from fastapi import APIRouter
from .tasks import tasks_router
from .metamodel import metamodel_router
from .metrics import metrics_router
from .training_routes import training_router
from .ssl import ssl_router
from .anomaly import anomaly_router
from .export import export_router
from .search import search_router
from .verify import verify_router

smartstable_router = APIRouter()
smartstable_router.include_router(tasks_router)
smartstable_router.include_router(metamodel_router)
smartstable_router.include_router(metrics_router)
smartstable_router.include_router(training_router)
smartstable_router.include_router(ssl_router)
smartstable_router.include_router(anomaly_router)
smartstable_router.include_router(export_router)
smartstable_router.include_router(search_router)
smartstable_router.include_router(verify_router)
