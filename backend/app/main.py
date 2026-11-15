import json, os, uuid, logging
from pathlib import Path
import joblib
try:
    import mlflow
    import mlflow.sklearn
except ImportError:  # API and training remain usable without optional tracking dependency.
    mlflow = None
import pandas as pd
from fastapi import FastAPI, Depends, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text
from .db import Base, engine, get_db, SessionLocal
from .entities import Dataset, Experiment, ModelRecord
from .schemas import LoginRequest, ExperimentRequest, PredictionRequest
from .core import settings
from .automl import train_all, detect_task
from .data_loader import load_csv, DatasetLoadError
from .data_validator import profile_dataset, validate_target

logger = logging.getLogger("automl.api")
Base.metadata.create_all(bind=engine)
app = FastAPI(title="AutoML Forge API", version="1.1.0",
              description="End-to-end AutoML lifecycle: upload, train, compare, register and predict.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

def dataset_frame(ds):
    return pd.read_csv(ds.path)

@app.get("/")
def root():
    return {"name":"AutoML Forge API","docs":"/docs","status":"online"}

@app.get("/api/health")
def health(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        logger.exception("Database health check failed")
        raise HTTPException(status_code=503, detail="Database is unavailable.") from exc
    return {"status": "healthy", "database": "connected"}

@app.post("/api/auth/login")
def login(payload: LoginRequest):
    if payload.email != "demo@automl.local" or payload.password != "demo123":
        raise HTTPException(401, "Invalid demo credentials. Use demo@automl.local / demo123.")
    return {"access_token":"demo-session-token","token_type":"bearer",
            "user":{"email":payload.email,"name":"Demo User","role":"admin"}}

@app.get("/api/dashboard")
def dashboard(db: Session = Depends(get_db)):
    return {"datasets":db.query(Dataset).count(), "experiments":db.query(Experiment).count(),
            "models":db.query(ModelRecord).count(),
            "deployed":db.query(ModelRecord).count(),
            "recent":[{"id":e.id,"target":e.target,"status":e.status,"task_type":e.task_type}
                      for e in db.query(Experiment).order_by(Experiment.id.desc()).limit(5)]}

@app.post("/api/datasets")
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only .csv files are accepted.")
    max_bytes = settings.max_upload_mb * 1024 * 1024
    content = await file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit.")
    path = Path(settings.upload_dir) / f"{uuid.uuid4().hex}.csv"
    path.write_bytes(content)
    try:
        frame = load_csv(path)
    except DatasetLoadError as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(400, f"Unable to read CSV: {exc}") from exc
    ds = Dataset(name=Path(file.filename).stem, filename=file.filename, path=str(path),
                 rows=len(frame), columns=len(frame.columns))
    db.add(ds); db.commit(); db.refresh(ds)
    return {"id":ds.id,"name":ds.name,"filename":ds.filename,"rows":ds.rows,"columns":ds.columns}

@app.get("/api/datasets")
def list_datasets(db: Session = Depends(get_db)):
    return [{"id":d.id,"name":d.name,"filename":d.filename,"rows":d.rows,"columns":d.columns,
             "created_at":d.created_at} for d in db.query(Dataset).order_by(Dataset.id.desc()).all()]

@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    ds = db.get(Dataset,dataset_id)
    if not ds: raise HTTPException(404,"Dataset not found.")
    df = dataset_frame(ds)
    profile = profile_dataset(df)
    return {"id":ds.id,"name":ds.name,"rows":profile["rows"],"columns":profile["columns"],
        "column_names":profile["column_names"],"dtypes":profile["dtypes"],
        "missing":profile["missing"],"duplicates":profile["duplicates"],
        "empty_columns":profile["empty_columns"],
        "preview":json.loads(df.head(10).to_json(orient="records",date_format="iso")),
        "statistics":json.loads(df.describe(include="all").fillna("").to_json())}

@app.post("/api/datasets/{dataset_id}/analyze")
def analyze_dataset(dataset_id: int, db: Session = Depends(get_db)):
    ds=db.get(Dataset,dataset_id)
    if not ds: raise HTTPException(404,"Dataset not found.")
    df=dataset_frame(ds)
    return {"numeric":df.select_dtypes(include="number").columns.tolist(),
            "categorical":df.select_dtypes(exclude="number").columns.tolist(),
            "missing_percent":(df.isna().mean()*100).round(2).to_dict(),
            "duplicates":int(df.duplicated().sum())}

def run_training(experiment_id: int):
    db=SessionLocal()
    try:
        exp=db.get(Experiment,experiment_id); ds=db.get(Dataset,exp.dataset_id)
        exp.status="running"; db.commit()
        cfg=json.loads(exp.config)
        if mlflow is not None:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(settings.mlflow_experiment_name)
        else:
            logger.warning("MLflow is not installed; experiment %s will not be tracked.", exp.id)
        result=train_all(dataset_frame(ds),exp.target,cfg["excluded_columns"],exp.task_type,
                         cfg["test_size"],cfg["cv_folds"],cfg.get("selection_metric"),
                         cfg["tune"],cfg.get("algorithms"),settings.model_dir,
                         cfg.get("random_state", 42))
        model_map={}
        for item in result["models"]:
            if "error" in item: continue
            run_id=None
            if mlflow is not None:
                try:
                    with mlflow.start_run(run_name=item["algorithm"]) as run:
                        run_id=run.info.run_id
                        mlflow.log_param("dataset_id",ds.id)
                        mlflow.log_param("target",exp.target)
                        mlflow.log_param("task_type",result["task_type"])
                        mlflow.log_param("algorithm",item["algorithm"])
                        mlflow.log_metric("cv_score",item["cv_score"])
                        for k,v in item["metrics"].items():
                            if isinstance(v,(int,float)): mlflow.log_metric(k,v)
                        mlflow.log_metric("training_duration_seconds",item["duration_seconds"])
                        mlflow.sklearn.log_model(joblib.load(item["artifact_path"]),"model")
                except Exception:
                    # Model training remains usable when the tracking service is temporarily
                    # unavailable, but the failure must be visible in server logs.
                    logger.exception("MLflow logging failed for experiment %s model %s", exp.id, item["algorithm"])
            rec=ModelRecord(experiment_id=exp.id,name=item["name"],algorithm=item["algorithm"],
                task_type=result["task_type"],metrics=json.dumps(item["metrics"]),cv_score=item["cv_score"],
                duration_seconds=item["duration_seconds"],artifact_path=item["artifact_path"],
                feature_names=json.dumps(item["feature_names"]),target_name=item["target_name"],mlflow_run_id=run_id)
            db.add(rec); db.flush(); model_map[item["id"]]=rec.id
        # Save a machine-readable comparison report and chart, then attach both to MLflow.
        successful = [item for item in result["models"] if "error" not in item]
        if successful:
            try:
                report_dir = Path(settings.model_dir) / "reports"
                report_dir.mkdir(parents=True, exist_ok=True)
                report_rows = []
                for item in successful:
                    report_rows.append({
                        "algorithm": item["algorithm"],
                        "cv_score": item["cv_score"],
                        "training_duration_seconds": item["duration_seconds"],
                        **{f"test_{key}": value for key, value in item["metrics"].items()
                           if isinstance(value, (int, float))},
                    })
                report_path = report_dir / f"experiment_{exp.id}_comparison.csv"
                pd.DataFrame(report_rows).to_csv(report_path, index=False)
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                metric_key = "f1" if result["task_type"] == "classification" else "rmse"
                chart_path = report_dir / f"experiment_{exp.id}_comparison.png"
                chart_frame = pd.DataFrame(report_rows).dropna(subset=[f"test_{metric_key}"])
                if not chart_frame.empty:
                    fig, ax = plt.subplots(figsize=(9, 4))
                    ax.bar(chart_frame["algorithm"], chart_frame[f"test_{metric_key}"])
                    ax.set_ylabel(f"Test {metric_key}")
                    ax.tick_params(axis="x", rotation=25)
                    fig.tight_layout()
                    fig.savefig(chart_path)
                    plt.close(fig)
                if mlflow is not None:
                    try:
                        with mlflow.start_run(run_name="model-comparison"):
                            mlflow.log_artifact(str(report_path))
                            if chart_path.exists():
                                mlflow.log_artifact(str(chart_path))
                    except Exception:
                        logger.exception("MLflow comparison artifact logging failed for experiment %s", exp.id)
            except Exception:
                logger.exception("Could not generate comparison artifacts for experiment %s", exp.id)
        exp.best_model_id=model_map.get(result["best_model_id"])
        exp.status="completed"; exp.task_type=result["task_type"]; db.commit()
        if exp.best_model_id and mlflow is not None:
            best=db.get(ModelRecord,exp.best_model_id)
            try:
                client=mlflow.tracking.MlflowClient()
                if best.mlflow_run_id:
                    registered_name = f"automl_dataset_{ds.id}_experiment_{exp.id}"
                    try:
                        client.get_registered_model(registered_name)
                    except Exception:
                        client.create_registered_model(registered_name)
                    mv=client.create_model_version(name=registered_name,
                        source=f"runs:/{best.mlflow_run_id}/model",run_id=best.mlflow_run_id)
                    best.registered_version=str(mv.version); db.commit()
            except Exception:
                logger.exception("MLflow model registration failed for experiment %s", exp.id)
    except Exception as exc:
        exp=db.get(Experiment,experiment_id)
        if exp: exp.status="failed"; exp.error=str(exc); db.commit()
    finally:
        db.close()

@app.post("/api/experiments")
def create_experiment(payload: ExperimentRequest, background: BackgroundTasks, db: Session=Depends(get_db)):
    ds=db.get(Dataset,payload.dataset_id)
    if not ds: raise HTTPException(404,"Dataset not found.")
    df=dataset_frame(ds)
    try:
        validate_target(df, payload.target)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    task=detect_task(df[payload.target]) if payload.task_type=="auto" else payload.task_type
    if task not in ("classification","regression"): raise HTTPException(400,"Invalid task type.")
    exp=Experiment(dataset_id=ds.id,target=payload.target,task_type=task,status="queued",
        config=json.dumps({"excluded_columns":payload.excluded_columns,"test_size":payload.test_size,
        "cv_folds":payload.cv_folds,"random_state":payload.random_state,
        "selection_metric":payload.selection_metric,"tune":payload.tune,
        "algorithms":payload.algorithms}))
    db.add(exp); db.commit(); db.refresh(exp)
    background.add_task(run_training,exp.id)
    return {"id":exp.id,"status":exp.status,"task_type":task}

@app.get("/api/experiments")
def list_experiments(db: Session=Depends(get_db)):
    return [{"id":e.id,"dataset_id":e.dataset_id,"target":e.target,"task_type":e.task_type,
             "status":e.status,"best_model_id":e.best_model_id,"error":e.error} for e in db.query(Experiment).order_by(Experiment.id.desc()).all()]

@app.get("/api/experiments/{experiment_id}")
def get_experiment(experiment_id:int,db:Session=Depends(get_db)):
    e=db.get(Experiment,experiment_id)
    if not e: raise HTTPException(404,"Experiment not found.")
    return {"id":e.id,"dataset_id":e.dataset_id,"target":e.target,"task_type":e.task_type,
            "status":e.status,"best_model_id":e.best_model_id,"error":e.error}

@app.get("/api/experiments/{experiment_id}/models")
def experiment_models(experiment_id:int,db:Session=Depends(get_db)):
    experiment = db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(404, "Experiment not found.")
    return [model_json(m) for m in db.query(ModelRecord).filter_by(experiment_id=experiment_id).all()]

def model_json(m):
    return {"id":m.id,"experiment_id":m.experiment_id,"name":m.name,"algorithm":m.algorithm,
        "task_type":m.task_type,"metrics":json.loads(m.metrics),"cv_score":m.cv_score,
        "duration_seconds":m.duration_seconds,"target_name":m.target_name,
        "registered_version":m.registered_version,"mlflow_run_id":m.mlflow_run_id}

@app.get("/api/models")
def list_models(db:Session=Depends(get_db)):
    return [model_json(m) for m in db.query(ModelRecord).order_by(ModelRecord.id.desc()).all()]

@app.get("/api/models/{model_id}")
def model_info(model_id:int,db:Session=Depends(get_db)):
    m=db.get(ModelRecord,model_id)
    if not m: raise HTTPException(404,"Model not found.")
    return {**model_json(m),"features":json.loads(m.feature_names)}

@app.post("/api/models/{model_id}/predict")
def predict(model_id:int,payload:PredictionRequest,db:Session=Depends(get_db)):
    m=db.get(ModelRecord,model_id)
    if not m: raise HTTPException(404,"Model not found.")
    if not payload.rows: raise HTTPException(400,"Provide at least one prediction row.")
    features=json.loads(m.feature_names)
    frame=pd.DataFrame(payload.rows)
    missing=set(features)-set(frame.columns)
    if missing: raise HTTPException(422,f"Missing required features: {', '.join(sorted(missing))}")
    try:
        estimator=joblib.load(m.artifact_path)
        predictions=estimator.predict(frame[features]).tolist()
        probabilities=estimator.predict_proba(frame[features]).tolist() if hasattr(estimator,"predict_proba") else None
        return {"model_id":m.id,"predictions":predictions,"probabilities":probabilities}
    except Exception as exc:
        raise HTTPException(400,f"Prediction failed: {exc}")


@app.get("/health")
def public_health():
    """Container liveness endpoint; database readiness remains available at /api/health."""
    return {"status": "healthy", "service": "AutoML Forge API"}

@app.get("/model-info")
def default_model_info(db: Session = Depends(get_db)):
    """Return metadata for the most recently selected best model."""
    record = db.query(ModelRecord).join(
        Experiment, ModelRecord.experiment_id == Experiment.id
    ).filter(Experiment.best_model_id == ModelRecord.id).order_by(ModelRecord.id.desc()).first()
    if record is None:
        raise HTTPException(404, "No selected model is available yet.")
    return {**model_json(record), "features": json.loads(record.feature_names),
            "artifact_exists": Path(record.artifact_path).is_file()}

@app.post("/predict")
def predict_with_best_model(payload: PredictionRequest, db: Session = Depends(get_db)):
    """Predict with the latest selected model using its saved preprocessing pipeline."""
    record = db.query(ModelRecord).join(
        Experiment, ModelRecord.experiment_id == Experiment.id
    ).filter(Experiment.best_model_id == ModelRecord.id).order_by(ModelRecord.id.desc()).first()
    if record is None:
        raise HTTPException(404, "No selected model is available for prediction.")
    if not payload.rows:
        raise HTTPException(400, "Provide at least one prediction row.")
    features = json.loads(record.feature_names)
    frame = pd.DataFrame(payload.rows)
    missing = set(features) - set(frame.columns)
    if missing:
        raise HTTPException(422, f"Missing required features: {', '.join(sorted(missing))}")
    try:
        if not Path(record.artifact_path).is_file():
            raise FileNotFoundError("Saved model artifact is missing.")
        estimator = joblib.load(record.artifact_path)
        predictions = estimator.predict(frame[features]).tolist()
        probabilities = estimator.predict_proba(frame[features]).tolist() if hasattr(estimator, "predict_proba") else None
        return {"model_id": record.id, "task_type": record.task_type,
                "predictions": predictions, "probabilities": probabilities}
    except Exception as exc:
        logger.exception("Default model prediction failed")
        raise HTTPException(500, f"Prediction failed: {exc}") from exc
