"""
train.py — Entrenamiento con SageMaker XGBoost built-in (SDK 2.257.3)
Usa los mejores hiperparámetros encontrados por Random Search en
preprocessing.py para lanzar el training job en SageMaker.
"""

import json
import boto3
import sagemaker
from sagemaker.estimator     import Estimator
from sagemaker.inputs        import TrainingInput
from sagemaker.serializers   import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
import config
from cw_logger import get_logger

logger = get_logger(__name__, stream_name="training")

#configuración de sesion
def get_session() -> sagemaker.Session:
    return sagemaker.Session(
        boto_session=boto3.Session(region_name=config.REGION)
    )

def _build_hyperparams(best_params: dict) -> dict:
    """
    Convierte los hiperparámetros encontrados al formato que XGBoost lee.
    """
    # Mapeo de nombres sklearn → SageMaker
    rename = {
        "n_estimators"    : "num_round",
        "learning_rate"   : "eta",
        "reg_lambda"      : "lambda",
        "reg_alpha"       : "alpha",
    }
    hp = {}
    for k, v in best_params.items():
        sm_key = rename.get(k, k)
        hp[sm_key] = v

    # parámetros fijos
    hp["objective"]             = "reg:squarederror"
    hp["eval_metric"]           = "rmse"
    hp["early_stopping_rounds"] = 30

    logger.info(f"Hiperparámetros para SageMaker {json.dumps(hp)}")
    return hp

# función de entrenamiento
def run_training(
    train_uri   : str,
    val_uri     : str,
    best_params : dict = None,
) -> Estimator:
    """
    Lanza el training job de SageMaker con XGBoost built-in.
    """
    logger.info("Inicio del training job")

    if best_params is None:
        logger.info("Cargando hiperparámetros desde S3")
        s3  = boto3.client("s3", region_name=config.REGION)
        obj = s3.get_object(
            Bucket = config.BUCKET_NAME,
            Key    = config.RS_RESULTS_S3_KEY,
        )
        rs_results  = json.loads(obj["Body"].read())
        best_params = rs_results.get("best_params", {})
        logger.info(f"Hiperparámetros cargados: {best_params}")

    hyperparams = _build_hyperparams(best_params)

    session   = get_session()
    image_uri = sagemaker.image_uris.retrieve(
        framework = "xgboost",
        region    = config.REGION,
        version   = config.XGBOOST_VERSION,
    )

    logger.info(f"Imagen XGBoost: {image_uri}")
    logger.info(f"Instancia: {config.TRAINING_INSTANCE}")
    logger.info(f"Train URI: {train_uri}")
    logger.info(f"Val URI  : {val_uri}")

    print(f"\n  Hiperparámetros:")
    for k, v in sorted(hyperparams.items()):
        print(f"    {k:<25}: {v}")

    estimator = Estimator(
        image_uri         = image_uri,
        role              = config.ROLE_ARN,
        instance_count    = 1,
        instance_type     = config.TRAINING_INSTANCE,
        output_path       = config.MODEL_S3_PATH,
        sagemaker_session = session,
        base_job_name     = config.JOB_BASE_NAME,
    )
    estimator.set_hyperparameters(**hyperparams)

    logger.info("Lanzando estimator.fit()")

    estimator.fit(
        inputs = {
            "train"     : TrainingInput(train_uri, content_type="text/csv"),
            "validation": TrainingInput(val_uri,   content_type="text/csv"),
        },
        logs = "Training",
    )

    job_name  = estimator.latest_training_job.name
    model_uri = estimator.model_data

    logger.info(f"Training completado Job: {job_name}")
    logger.info(f"Artefacto modelo {model_uri}")
    return estimator


#deployea el endpoint en tiempo real
def deploy_endpoint(
    estimator     : Estimator,
    endpoint_name : str = config.ENDPOINT_NAME,
) -> sagemaker.predictor.Predictor:
    """
    Despliega el modelo como endpoint de inferencia en tiempo real.
    """
    logger.info(f"Desplegando endpoint: {endpoint_name}")

    predictor = estimator.deploy(
        initial_instance_count = 1,
        instance_type          = config.ENDPOINT_INSTANCE,
        endpoint_name          = endpoint_name,
        serializer             = CSVSerializer(),
        deserializer           = JSONDeserializer(),
    )

    logger.info(f"Endpoint activo: {endpoint_name}")
    return predictor

# elimina el endpoint para ahorrar dinero
def delete_endpoint(endpoint_name: str = config.ENDPOINT_NAME) -> None:
    """
    Borra el endpoint de sagemaker
    """
    boto3.client("sagemaker", region_name=config.REGION) \
         .delete_endpoint(EndpointName=endpoint_name)
    logger.info(f"Endpoint eliminado: {endpoint_name}")
    
# lista los training jobs
def list_training_jobs(max_results: int = 5) -> None:
    """
    Se entrenaron varios modelos con distintos hiperamtros
    de busqueda para encontrar el mejor.
    Por lo anterior, era necesario listar los training jobs
    """
    sm   = boto3.client("sagemaker", region_name=config.REGION)
    jobs = sm.list_training_jobs(
        NameContains = config.JOB_BASE_NAME,
        MaxResults   = max_results,
        SortBy       = "CreationTime",
        SortOrder    = "Descending",
    )["TrainingJobSummaries"]
    for j in jobs:
        status = j["TrainingJobStatus"]
        logger.info(f"Job: {j['TrainingJobName']}  Status: {status}")
        
# levanta el endpoint
def attach_existing_endpoint(
    endpoint_name: str = config.ENDPOINT_NAME,
) -> sagemaker.predictor.Predictor:
    """
    Reconecta el endpoint.
    """
    session = get_session()
    predictor = sagemaker.predictor.Predictor(
        endpoint_name     = endpoint_name,
        sagemaker_session = session,
        serializer        = CSVSerializer(),
        deserializer      = JSONDeserializer(),
    )
    logger.info(f"Reconectado al endpoint: {endpoint_name}")
    return predictor

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        sys.exit(1)
    estimator = run_training(sys.argv[1], sys.argv[2])
    deploy_endpoint(estimator)
