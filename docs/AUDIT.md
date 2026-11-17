# Initial audit and implementation report

> **Historical document.** This audit was written while login was still a demo-only stub.
> That is no longer true: the API now uses PBKDF2 password hashing, signed JWT tokens,
> protected routes and admin/user roles (see `backend/app/auth.py` and `backend/tests/test_auth.py`).
> Any line below that calls authentication a "demo stub" or "demo gate" is **outdated** and kept
> only as a record of how the project evolved. The current feature list is in the README.

## Scope inspected

The supplied archive contains 31 project files. The audit covered the frontend source and build configuration, FastAPI routes, SQLAlchemy entities and database setup, ML pipeline, schemas and settings, tests, requirements, Dockerfiles, Compose configuration, DVC pipeline, GitHub Actions workflow, sample data, README, and cloud notes.

## Existing implementation

- React/Vite/Tailwind dashboard with Overview, Datasets, Experiments, and Models pages.
- FastAPI routes for demo login, dashboard, CSV upload, dataset profiling, experiments, model listing, model information, and predictions.
- SQLAlchemy metadata for datasets, experiments, and models.
- CSV persistence and basic data validation.
- scikit-learn preprocessing pipelines, train/test split, cross-validation, classification/regression metrics, and joblib serialization.
- MLflow tracking and best-model registration attempts.
- Docker Compose services for PostgreSQL, MLflow PostgreSQL, MLflow, API, and frontend.
- A DVC dataset-validation stage and a small pytest suite.

## Findings

### Critical / high priority

1. Demo authentication is hardcoded and is not checked by protected API dependencies. There is no JWT verification or user isolation.
2. CORS originally allowed every origin; it is now configurable and defaults to the local Vite origin.
3. Numeric task detection misclassified a small numeric class-label sample. This was fixed and regression-tested.
4. MLflow integration exceptions were silently ignored. They now produce server-side exception logs while allowing model training to finish.
5. Dataset upload originally read the complete file into memory before enforcing the size limit. It now reads at most the configured limit plus one byte.
6. The original model catalog lacked Extra Trees and HistGradientBoosting, plus Lasso regression. These were added.
7. Cross-validation and preprocessing edge cases needed stronger validation and handling.
8. Docker MLflow artifact configuration used a local artifact root without sharing that volume with the API container. Compose now enables MLflow artifact serving with an artifacts destination volume.

### Medium priority / incomplete

- SQLAlchemy `create_all` is used instead of Alembic migrations.
- Background training is in-process, without a durable worker queue.
- No model/dataset ownership or role-based access control exists.
- The frontend does not send or verify an authorization token; its current login is only a demo gate.
- Dataset analysis is primarily tabular; interactive profiling and explainability are incomplete.
- DVC has no configured remote.
- CI does not yet run frontend component or browser workflow tests.
- The project has no drift monitoring, retraining scheduler, or downloadable reports.

## Changes made in this pass

- Reworked the AutoML pipeline while retaining its public `train_all`, `detect_task`, and `make_pipeline` entry points.
- Added Extra Trees and HistGradientBoosting for classification and regression, plus Lasso regression.
- Improved task detection, split/CV selection, all-missing column handling, metric validation, and model error reporting.
- Added randomized search for Random Forest and Extra Trees.
- Added six backend unit tests.
- Configured CORS origins rather than wildcard origins.
- Added a database connectivity check to the health endpoint.
- Made MLflow tracking and registration failures visible in logs.
- Bounded upload reads to the configured upload limit plus one byte.
- Fixed MLflow artifact serving configuration in Docker Compose.
- Updated README and added architecture/workflow diagrams.

## Verification results

- Backend pytest: **8 passed**.
- Frontend build: **not verified**. `npm run build` could not execute Vite because frontend dependencies are not installed in this environment; the command returned `vite: Permission denied`.
- Docker Compose validation: **not verified**. Docker is not executable/available in this environment.
- PostgreSQL and MLflow connectivity: **not verified** because the Compose services were not started.
- API integration against PostgreSQL: **not verified**.

## Prioritized remaining work

1. Replace demo login with a real identity model, password hashing, JWT access/refresh tokens, protected routes, and tenant scoping.
2. Add Alembic migrations and relationships/constraints for users, dataset versions, training jobs, model versions, and predictions.
3. Add durable worker queue and job progress/cancellation.
4. Add frontend tests and complete authenticated API integration.
5. Add richer data profiling, feature selection, imbalance handling, SHAP explanations, and evaluation report exports.
6. Complete MLflow registry lifecycle, DVC remote documentation/testing, prediction logging, drift detection, and retraining workflows.
7. Add container health checks for all relevant services, security/dependency scanning, and deployment automation.


## AutoML Forge integration upgrade (follow-up)

### Additional implementation

- Added reusable `data_loader`, `data_validator`, `task_detector`, and `preprocessing` modules.
- Added independent baseline `model_trainer`, task-aware `metrics`, `model_evaluator`, and metric-direction-aware `model_selector` modules.
- Added a best-effort `MLflowTracker` helper that returns structured failure information rather than interrupting training.
- Added standard `GET /health`, `GET /model-info`, and `POST /predict` routes. The default prediction route loads the selected model's complete saved preprocessing+estimator pipeline.
- Reused the shared CSV loader in the existing upload route and updated API branding to AutoML Forge.
- Added a reproducible DVC training stage using the bundled `purchased` target, generated model artifact, and metrics report.
- Added Docker startup and liveness health-check configuration.
- Expanded CI to compile Python sources, run backend tests, and build the API image; retained the existing React production build job.
- Extended backend tests for CSV errors, profiling, multiclass metrics, both model families, model evaluation, artifact selection, and API health.
- Updated frontend branding and surfaced MLflow run/registry information in model cards.

### Follow-up verification

- Backend pytest: **29 passed**.
- Python compile check: **passed** (`compileall`).
- DVC training-stage script: **passed**; trained eight models on the bundled sample and produced the DVC best-model artifact and metrics report.
- Synthetic end-to-end classification: **passed**; all four requested classifier families trained and produced test metrics.
- Synthetic end-to-end regression: **passed**; all four requested regressor families trained and produced test metrics.
- FastAPI smoke test with a persisted selected model: `/health` **200**, `/model-info` **200**, `/predict` **200**, missing feature input **422**.
- Docker image build and Compose startup: **not run** because Docker is unavailable in the execution environment.
- DVC CLI: **not run** because the DVC executable is unavailable in the execution environment; the validation and training stage scripts were executed directly, including the eight-model sample run.
- React production build: attempted, but Vite exited with `Permission denied`; the frontend build remains unverified.
- MLflow server/registry integration: **not end-to-end verified** because MLflow is not installed and no tracking server is running in the execution environment.

### Remaining production considerations

- ~~The existing login remains a demo-only stub~~ (outdated: JWT authentication was added in the Semester 2 pass below).
- Existing background tasks are in-process and not durable across restarts.
- SQLAlchemy schema changes are not managed by Alembic migrations.
- A DVC remote and MLflow tracking server/object-store credentials must be configured by the deployment owner.
- The existing dashboard is React/Vite rather than Streamlit. This was preserved to avoid replacing the working frontend; the required workflow is surfaced through the existing React interface and FastAPI service.

## Semester 2 implementation status

- Added PBKDF2 password hashing, signed JWT login/register/me endpoints, protected API routes, and admin/user role primitives.
- Added Alembic configuration plus user, dataset-version, and model-version metadata tables.
- Added Optuna tuning when installed, randomized-search fallback, optional percentile feature selection, and balanced class weights.
- Added dataset quality scoring, feature importance, ROC curve data, and regression residual data endpoints.
- Added authenticated frontend requests, training option controls, settings page, frontend smoke test, and a GitHub Actions CD workflow that publishes API/frontend images to GHCR.

## Final demo additions

- Analytics page: dataset quality scores, per-task algorithm performance, model comparison with filters, prediction history.
- Explainability page: feature importance chart (tree importance, coefficients, or permutation importance), confusion matrix and ROC curve for classification, residual plot for regression, downloadable Markdown evaluation report.
- Dataset quality score and dataset version shown on the dataset profile.
- Training progress percentage and current step shown while an experiment runs.
- Voting ensemble (soft voting where possible) built from the top three models of every run.
- Prediction history table and `/api/predictions` endpoint; model filtering on `/api/models`.
- MLflow UI link in the sidebar, settings, and model cards.
- New tests: `test_auth.py` (hashing, login, register, token expiry/tampering, roles) and `test_explain.py` (explanations, plots, report, ensemble, history).
- CI: Docker Compose smoke test and dependency security scanning.
