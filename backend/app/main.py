import json, os, uuid, logging
from pathlib import Path
import joblib
try:
    import mlflow
    import mlflow.sklearn
except ImportError:  # API and training remain usable without optional tracking dependency.
    mlflow = None
import pandas as pd
import numpy as np
from fastapi import FastAPI, Depends, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from .db import Base, engine, get_db, SessionLocal
from .entities import Dataset, Experiment, ModelRecord, User, DatasetVersion, ModelVersion, PredictionLog
from .schemas import LoginRequest, RegisterRequest, ExperimentRequest, PredictionRequest
from .core import settings
from .auth import create_access_token, get_current_user, hash_password, verify_password, require_role
from .automl import train_all, detect_task
from .data_loader import load_csv, DatasetLoadError
from .data_validator import profile_dataset, validate_target

logger = logging.getLogger("automl.api")
Base.metadata.create_all(bind=engine)
def ensure_demo_user():
    db = SessionLocal()
    try:
        if not db.query(User).filter_by(email="demo@automl.local").first():
            db.add(User(email="demo@automl.local", name="Demo User",
                        password_hash=hash_password("demo123"), role="admin"))
            db.commit()
    finally:
        db.close()

ensure_demo_user()
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


def quality_score(df):
    """0-100 score: penalise missing cells (up to 60 points) and duplicate rows (up to 30)."""
    if df.empty:
        return 0.0
    return max(0.0, round(100 - (float(df.isna().mean().mean()) * 60)
                          - (float(df.duplicated().mean()) * 30), 2))


def quality_label(score):
    return "good" if score >= 80 else "review" if score >= 60 else "poor"


# Live training progress (0-100) per experiment. Training runs in-process, so a simple
# dictionary is enough; finished experiments fall back to their stored status.
TRAINING_PROGRESS = {}


def experiment_progress(e):
    if e.status == "completed":
        return 100
    if e.status == "queued":
        return 0
    return TRAINING_PROGRESS.get(e.id, {}).get("percent", 0)


def experiment_step(e):
    if e.status in ("completed", "failed", "queued"):
        return e.status
    return TRAINING_PROGRESS.get(e.id, {}).get("step", "starting")


def experiment_json(e):
    return {"id":e.id,"dataset_id":e.dataset_id,"target":e.target,"task_type":e.task_type,
            "status":e.status,"best_model_id":e.best_model_id,"error":e.error,
            "progress":experiment_progress(e),"current_step":experiment_step(e)}

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
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter_by(email=payload.email.lower()).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(401, "Invalid email or password.")
    return {"access_token": create_access_token(user), "token_type": "bearer",
            "user":{"id": user.id, "email":user.email, "name":user.name, "role":user.role}}


@app.post("/api/auth/register")
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    email = payload.email.lower().strip()
    if db.query(User).filter_by(email=email).first():
        raise HTTPException(409, "An account with this email already exists.")
    user = User(email=email, name=payload.name.strip(), password_hash=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"access_token": create_access_token(user), "token_type": "bearer",
            "user": {"id": user.id, "email": user.email, "name": user.name, "role": user.role}}


@app.get("/api/auth/me")
def current_user(user: User = Depends(get_current_user)):
    return {"id": user.id, "email": user.email, "name": user.name, "role": user.role}


@app.get("/api/admin/users")
def list_users(db: Session = Depends(get_db), user: User = Depends(require_role("admin"))):
    return [{"id": item.id, "email": item.email, "name": item.name,
             "role": item.role, "is_active": item.is_active}
            for item in db.query(User).order_by(User.id).all()]

@app.get("/api/dashboard")
def dashboard(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return {"datasets":db.query(Dataset).count(), "experiments":db.query(Experiment).count(),
            "models":db.query(ModelRecord).count(),
            "deployed":db.query(ModelRecord).count(),
            "recent":[{"id":e.id,"target":e.target,"status":e.status,"task_type":e.task_type}
                      for e in db.query(Experiment).order_by(Experiment.id.desc()).limit(5)]}

@app.post("/api/datasets")
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db),
                         user: User = Depends(get_current_user)):
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
                 rows=len(frame), columns=len(frame.columns), owner=user.email)
    db.add(ds); db.commit(); db.refresh(ds)
    import hashlib
    db.add(DatasetVersion(dataset_id=ds.id, version=1,
                          checksum=hashlib.sha256(content).hexdigest(), path=str(path)))
    db.commit()
    return {"id":ds.id,"name":ds.name,"filename":ds.filename,"rows":ds.rows,"columns":ds.columns}

@app.get("/api/datasets")
def list_datasets(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return [{"id":d.id,"name":d.name,"filename":d.filename,"rows":d.rows,"columns":d.columns,
             "created_at":d.created_at} for d in db.query(Dataset).order_by(Dataset.id.desc()).all()]

@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    ds = db.get(Dataset,dataset_id)
    if not ds: raise HTTPException(404,"Dataset not found.")
    df = dataset_frame(ds)
    profile = profile_dataset(df)
    quality = quality_score(df)
    latest = db.query(DatasetVersion).filter_by(dataset_id=ds.id).order_by(DatasetVersion.version.desc()).first()
    return {"id":ds.id,"name":ds.name,"rows":profile["rows"],"columns":profile["columns"],
        "version": latest.version if latest else 1,
        "checksum": latest.checksum if latest else None,
        "quality_label": quality_label(quality),
        "column_names":profile["column_names"],"dtypes":profile["dtypes"],
        "missing":profile["missing"],"duplicates":profile["duplicates"],
        "quality_score": quality,
        "empty_columns":profile["empty_columns"],
        "preview":json.loads(df.head(10).to_json(orient="records",date_format="iso")),
        "statistics":json.loads(df.describe(include="all").fillna("").to_json())}

@app.post("/api/datasets/{dataset_id}/analyze")
def analyze_dataset(dataset_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    ds=db.get(Dataset,dataset_id)
    if not ds: raise HTTPException(404,"Dataset not found.")
    df=dataset_frame(ds)
    quality = quality_score(df)
    return {"numeric":df.select_dtypes(include="number").columns.tolist(),
            "categorical":df.select_dtypes(exclude="number").columns.tolist(),
            "missing_percent":(df.isna().mean()*100).round(2).to_dict(),
            "duplicates":int(df.duplicated().sum()), "quality_score": quality,
            "quality_label": quality_label(quality)}

def run_training(experiment_id: int):
    db=SessionLocal()
    try:
        exp=db.get(Experiment,experiment_id); ds=db.get(Dataset,exp.dataset_id)
        exp.status="running"; db.commit()
        TRAINING_PROGRESS[exp.id] = {"percent": 1, "step": "starting"}
        def on_progress(done, total, step):
            # Model fitting is the first 90%; saving and registration take the rest.
            TRAINING_PROGRESS[exp.id] = {"percent": int(5 + 85 * done / max(total, 1)), "step": step}
        cfg=json.loads(exp.config)
        if mlflow is not None:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(settings.mlflow_experiment_name)
        else:
            logger.warning("MLflow is not installed; experiment %s will not be tracked.", exp.id)
        result=train_all(dataset_frame(ds),exp.target,cfg["excluded_columns"],exp.task_type,
                         cfg["test_size"],cfg["cv_folds"],cfg.get("selection_metric"),
                         cfg["tune"],cfg.get("algorithms"),settings.model_dir,
                         cfg.get("random_state", 42), cfg.get("feature_selection", False),
                         cfg.get("balance_classes", False), cfg.get("ensemble", True),
                         on_progress)
        TRAINING_PROGRESS[exp.id] = {"percent": 92, "step": "saving models"}
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
            db.add(ModelVersion(model_id=rec.id, version=1, artifact_path=item["artifact_path"]))
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
        TRAINING_PROGRESS.pop(exp.id, None)
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
        db.rollback()
        exp=db.get(Experiment,experiment_id)
        if exp: exp.status="failed"; exp.error=str(exc); db.commit()
        TRAINING_PROGRESS.pop(experiment_id, None)
    finally:
        db.close()

@app.post("/api/experiments")
def create_experiment(payload: ExperimentRequest, background: BackgroundTasks, db: Session=Depends(get_db),
                      user: User = Depends(get_current_user)):
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
        "algorithms":payload.algorithms, "feature_selection": payload.feature_selection,
        "balance_classes": payload.balance_classes, "ensemble": payload.ensemble}))
    db.add(exp); db.commit(); db.refresh(exp)
    background.add_task(run_training,exp.id)
    return {"id":exp.id,"status":exp.status,"task_type":task,"progress":0}

@app.get("/api/experiments")
def list_experiments(db: Session=Depends(get_db), user: User = Depends(get_current_user)):
    return [experiment_json(e) for e in db.query(Experiment).order_by(Experiment.id.desc()).all()]

@app.get("/api/experiments/{experiment_id}")
def get_experiment(experiment_id:int,db:Session=Depends(get_db), user: User = Depends(get_current_user)):
    e=db.get(Experiment,experiment_id)
    if not e: raise HTTPException(404,"Experiment not found.")
    return experiment_json(e)

@app.get("/api/experiments/{experiment_id}/models")
def experiment_models(experiment_id:int,db:Session=Depends(get_db), user: User = Depends(get_current_user)):
    experiment = db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(404, "Experiment not found.")
    return [model_json(m) for m in db.query(ModelRecord).filter_by(experiment_id=experiment_id).all()]

def model_json(m, version=None):
    return {"id":m.id,"experiment_id":m.experiment_id,"name":m.name,"algorithm":m.algorithm,
        "task_type":m.task_type,"metrics":json.loads(m.metrics),"cv_score":m.cv_score,
        "duration_seconds":m.duration_seconds,"target_name":m.target_name,
        "registered_version":m.registered_version,"mlflow_run_id":m.mlflow_run_id,
        "version":version or 1,
        "created_at":m.created_at.isoformat() if m.created_at else None}


def model_versions(db):
    versions = {}
    for item in db.query(ModelVersion).all():
        versions[item.model_id] = max(item.version, versions.get(item.model_id, 0))
    return versions


def clean_feature_name(name):
    for prefix in ("numeric__", "categorical__", "num__", "cat__", "remainder__"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def explain_model(estimator, frame, feature_names):
    model = getattr(estimator, "named_steps", {}).get("model", estimator)
    preprocessor = getattr(estimator, "named_steps", {}).get("preprocessor")
    names = feature_names
    if preprocessor is not None:
        try:
            names = list(preprocessor.get_feature_names_out())
        except (AttributeError, ValueError):
            pass
    selector = getattr(estimator, "named_steps", {}).get("feature_selection")
    if selector is not None and hasattr(selector, "get_support"):
        names = [name for name, selected in zip(names, selector.get_support()) if selected]
    values = getattr(model, "feature_importances_", None)
    if values is None and hasattr(model, "coef_"):
        values = np.abs(model.coef_).mean(axis=0) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
    if values is None:
        return {"method": "unavailable", "feature_importance": []}
    pairs = sorted(zip([clean_feature_name(str(n)) for n in names], [float(value) for value in values]),
                   key=lambda item: item[1], reverse=True)[:15]
    return {"method": "model coefficients" if hasattr(model, "coef_") else "tree importance",
            "feature_importance": [{"feature": name, "importance": value} for name, value in pairs]}

def permutation_explanation(estimator, frame, target, task_type, features, max_rows=300):
    """Model-agnostic importance for models without coefficients/tree importances
    (for example the voting ensemble, KNN or SVM)."""
    from sklearn.inspection import permutation_importance
    data = frame.dropna(subset=[target])
    if len(data) > max_rows:
        data = data.sample(max_rows, random_state=42)
    result = permutation_importance(estimator, data[features], data[target], n_repeats=3,
                                    random_state=42,
                                    scoring="f1_weighted" if task_type == "classification" else "r2")
    pairs = sorted(zip(features, [float(v) for v in result.importances_mean]),
                   key=lambda item: item[1], reverse=True)[:15]
    return {"method": "permutation importance",
            "feature_importance": [{"feature": name, "importance": value} for name, value in pairs]}


def build_explanation(record, db):
    estimator = joblib.load(record.artifact_path)
    features = json.loads(record.feature_names)
    explanation = explain_model(estimator, pd.DataFrame(), features)
    if not explanation["feature_importance"]:
        exp = db.get(Experiment, record.experiment_id)
        ds = db.get(Dataset, exp.dataset_id) if exp else None
        if ds and Path(ds.path).is_file():
            try:
                explanation = permutation_explanation(estimator, dataset_frame(ds), record.target_name,
                                                      record.task_type, features)
            except Exception:
                logger.exception("Permutation importance failed for model %s", record.id)
    metrics = {}
    try:
        metrics = json.loads(record.metrics)
    except (TypeError, ValueError):
        logger.warning("Stored metrics for model %s are not valid JSON", record.id)
    explanation.update({
        "model_id": record.id, "algorithm": record.algorithm, "task_type": record.task_type,
        "target": record.target_name, "cv_score": record.cv_score,
        "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        "plots": {
            "roc_curve": metrics.get("roc_curve"),
            "residual_plot": metrics.get("residuals"),
            "confusion_matrix": metrics.get("confusion_matrix"),
            "labels": metrics.get("labels"),
        },
    })
    try:
        import shap  # noqa: F401  Optional: install shap when local SHAP plots are required.
        explanation["shap_available"] = True
    except ImportError:
        explanation["shap_available"] = False
    return explanation


@app.get("/api/config")
def ui_config(user: User = Depends(get_current_user)):
    return {"mlflow_url": settings.mlflow_ui_url}


@app.get("/api/models")
def list_models(task_type: str | None = None, algorithm: str | None = None,
                experiment_id: int | None = None,
                db:Session=Depends(get_db), user: User = Depends(get_current_user)):
    query = db.query(ModelRecord)
    if task_type:
        query = query.filter(ModelRecord.task_type == task_type)
    if algorithm:
        query = query.filter(ModelRecord.algorithm == algorithm)
    if experiment_id:
        query = query.filter(ModelRecord.experiment_id == experiment_id)
    versions = model_versions(db)
    return [model_json(m, versions.get(m.id)) for m in query.order_by(ModelRecord.id.desc()).all()]

@app.get("/api/models/{model_id}")
def model_info(model_id:int,db:Session=Depends(get_db), user: User = Depends(get_current_user)):
    m=db.get(ModelRecord,model_id)
    if not m: raise HTTPException(404,"Model not found.")
    return {**model_json(m, model_versions(db).get(m.id)),"features":json.loads(m.feature_names)}


@app.get("/api/models/{model_id}/explain")
def explain(model_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    record = db.get(ModelRecord, model_id)
    if not record:
        raise HTTPException(404, "Model not found.")
    try:
        return build_explanation(record, db)
    except Exception as exc:
        logger.exception("Model explanation failed")
        raise HTTPException(500, "Could not generate model explanation.") from exc


@app.get("/api/models/{model_id}/report", response_class=PlainTextResponse)
def evaluation_report(model_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Downloadable Markdown evaluation report for one model."""
    record = db.get(ModelRecord, model_id)
    if not record:
        raise HTTPException(404, "Model not found.")
    try:
        info = build_explanation(record, db)
    except Exception as exc:
        logger.exception("Report generation failed")
        raise HTTPException(500, "Could not generate evaluation report.") from exc
    lines = [f"# Evaluation report: {record.algorithm} (model #{record.id})", "",
             f"- Experiment: #{record.experiment_id}",
             f"- Task: {record.task_type}",
             f"- Target: {record.target_name}",
             f"- Cross-validation score: {record.cv_score:.4f}" if record.cv_score is not None else "- Cross-validation score: n/a",
             f"- Training time: {record.duration_seconds:.2f}s",
             f"- MLflow run: {record.mlflow_run_id or 'not tracked'}", "",
             "## Test metrics", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {v:.4f} |" for k, v in info["metrics"].items()]
    lines += ["", f"## Feature importance ({info['method']})", "", "| Feature | Importance |", "|---|---|"]
    lines += [f"| {f['feature']} | {f['importance']:.4f} |" for f in info["feature_importance"][:10]]
    cm, labels = info["plots"].get("confusion_matrix"), info["plots"].get("labels")
    if cm and labels:
        lines += ["", "## Confusion matrix (rows = actual, columns = predicted)", "",
                  "| | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
        lines += [f"| {labels[i]} | " + " | ".join(str(v) for v in row) + " |" for i, row in enumerate(cm)]
    return PlainTextResponse("\n".join(lines) + "\n", headers={
        "Content-Disposition": f'attachment; filename="model_{record.id}_report.md"'})


@app.get("/api/analytics")
def analytics(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Workspace-level numbers for the Analytics page."""
    datasets = []
    for ds in db.query(Dataset).order_by(Dataset.id.desc()).all():
        try:
            score = quality_score(dataset_frame(ds))
        except Exception:
            score = None
        latest = db.query(DatasetVersion).filter_by(dataset_id=ds.id).order_by(DatasetVersion.version.desc()).first()
        datasets.append({"id": ds.id, "name": ds.name, "rows": ds.rows, "columns": ds.columns,
                         "quality_score": score,
                         "quality_label": quality_label(score) if score is not None else "unreadable",
                         "version": latest.version if latest else 1})
    status_counts = {}
    for e in db.query(Experiment).all():
        status_counts[e.status] = status_counts.get(e.status, 0) + 1
    algorithms = {}
    for m in db.query(ModelRecord).all():
        entry = algorithms.setdefault((m.task_type, m.algorithm), {"algorithm": m.algorithm, "task_type": m.task_type,
                                                                    "count": 0, "scores": [], "wins": 0})
        entry["count"] += 1
        if m.cv_score is not None:
            entry["scores"].append(m.cv_score)
    for e in db.query(Experiment).filter(Experiment.best_model_id.isnot(None)).all():
        best = db.get(ModelRecord, e.best_model_id)
        if best and (best.task_type, best.algorithm) in algorithms:
            algorithms[(best.task_type, best.algorithm)]["wins"] += 1
    # CV scores use different metrics per task (weighted F1 vs negative RMSE), so rows are per task.
    algorithm_rows = [{"algorithm": a["algorithm"], "task_type": a["task_type"], "count": a["count"], "wins": a["wins"],
                       "avg_cv_score": round(sum(a["scores"]) / len(a["scores"]), 4) if a["scores"] else None}
                      for a in algorithms.values()]
    return {"datasets": datasets, "experiments_by_status": status_counts,
            "algorithms": sorted(algorithm_rows, key=lambda r: r["count"], reverse=True),
            "predictions": db.query(PredictionLog).count()}


def log_prediction(db, model_id, user_email, rows, predictions):
    try:
        db.add(PredictionLog(model_id=model_id, user_email=user_email,
                             inputs=json.dumps(rows, default=str),
                             outputs=json.dumps(predictions, default=str)))
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Could not store prediction history")


@app.get("/api/predictions")
def prediction_history(model_id: int | None = None, limit: int = 50,
                       db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    query = db.query(PredictionLog)
    if model_id:
        query = query.filter(PredictionLog.model_id == model_id)
    return [{"id": p.id, "model_id": p.model_id, "user": p.user_email,
             "inputs": json.loads(p.inputs), "outputs": json.loads(p.outputs),
             "created_at": p.created_at.isoformat() if p.created_at else None}
            for p in query.order_by(PredictionLog.id.desc()).limit(min(limit, 200)).all()]

@app.post("/api/models/{model_id}/predict")
def predict(model_id:int,payload:PredictionRequest,db:Session=Depends(get_db),
            user: User = Depends(get_current_user)):
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
        log_prediction(db, m.id, user.email, payload.rows, predictions)
        return {"model_id":m.id,"predictions":predictions,"probabilities":probabilities,
                "explanation": explain_model(estimator, frame[features], features)}
    except Exception as exc:
        raise HTTPException(400,f"Prediction failed: {exc}")


@app.get("/health")
def public_health():
    """Container liveness endpoint; database readiness remains available at /api/health."""
    return {"status": "healthy", "service": "AutoML Forge API"}

@app.get("/model-info")
def default_model_info(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
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
        log_prediction(db, record.id, None, payload.rows, predictions)
        return {"model_id": record.id, "task_type": record.task_type,
                "predictions": predictions, "probabilities": probabilities}
    except Exception as exc:
        logger.exception("Default model prediction failed")
        raise HTTPException(500, f"Prediction failed: {exc}") from exc
