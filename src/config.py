"""
config.py Configuración compartida
"""
import boto3
import sagemaker

# Configuración de AWS
BUCKET_NAME = "airbnb-pred"       # ← tu bucket
PREFIX      = "airbnb-price"
REGION      = "us-east-1"          # ← tu región

try:
    ROLE_ARN = sagemaker.get_execution_role()
except Exception:
    ROLE_ARN = "aws:iam::141095608224:role/SageMakerStudioExecutionRole2026"

ACCOUNT_ID = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]

# Configuración de SNS
SNS_TOPIC_NAME_NOVATO  = "TopicNovato"
SNS_TOPIC_NAME_EXPERTO = "TopicExperto"
SNS_TOPIC_ARN_NOVATO  = f"arn:aws:sns:{REGION}:{ACCOUNT_ID}:{SNS_TOPIC_NAME_NOVATO}"
SNS_TOPIC_ARN_EXPERTO = f"arn:aws:sns:{REGION}:{ACCOUNT_ID}:{SNS_TOPIC_NAME_EXPERTO}"
EMAIL_NOVATO  = "raulpopoca73@gmail.com"   # sin propiedad, quiere comprar/invertir
EMAIL_EXPERTO = "dianaba25@gmail.com"   # ya tiene propiedad, quiere info especializada

# Datos
RAW_PARQUET_KEY      = "airbnb_final.parquet"
RAW_S3_URI           = f"s3://{BUCKET_NAME}/{PREFIX}/raw/{RAW_PARQUET_KEY}"
TRAIN_S3_URI         = f"s3://{BUCKET_NAME}/{PREFIX}/data/train/train.csv"
VAL_S3_URI           = f"s3://{BUCKET_NAME}/{PREFIX}/data/validation/validation.csv"
ENCODINGS_S3_KEY     = f"{PREFIX}/artifacts/encodings.json"
RS_RESULTS_S3_KEY    = f"{PREFIX}/artifacts/random_search_results.json"
GEO_CACHE_S3_KEY     = f"{PREFIX}/artifacts/geo_cache.json"

# Endpoint de inferencia
MODEL_S3_PATH  = f"s3://{BUCKET_NAME}/{PREFIX}/model"
ENDPOINT_NAME  = "airbnb-v2-xgboost"
JOB_BASE_NAME  = "airbnb-v2-xgb"

# Grupo en CloudWatch
CW_LOG_GROUP   = "/airbnb-price-prediction/v2"

# Preprocessing
SMOOTHING_FACTOR       = 10   # suavizado del encoding
OVERSAMPLE_MIN_SAMPLES = 200  # umbral de oversampling

LAT_CENTER = 19.42740    # Ángel de la independencia
LON_CENTER = -99.16735

# Columnas ordenadas para hacer el training
FEATURE_COLS = [
    "log_price",
    "latitude", "longitude",
    "accommodates", "bathrooms", "bedrooms", "beds",
    "parking", "patio_balcon",
    "room_type_encoded",
    "guests_per_room", "beds_per_guest", "amenity_score",
    "neighborhood_median_price", "neighbourhood_encoded",
    "room_type_median_price",
    "geo_cluster_encoded", "capacity_bath",
    "dist_centro", "is_large", "is_private_room",
]
# se excluye log_price porque es el target
INFERENCE_COLS = [c for c in FEATURE_COLS if c != "log_price"]

# Random Search
RS_N_ITER      = 50      # número de combinaciones a probar
RS_CV_FOLDS    = 5       # folds de cross-validation
RS_SCORING     = "neg_root_mean_squared_error"
RS_RANDOM_SEED = 53

RS_PARAM_DISTRIBUTIONS = {
    "n_estimators"     : [200, 300, 400, 500, 600],
    "max_depth"        : [3, 4, 5, 6, 7, 8],
    "learning_rate"    : [0.01, 0.03, 0.05, 0.08, 0.1],
    "subsample"        : [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree" : [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight" : [1, 3, 5, 7, 10],
    "gamma"            : [0, 0.1, 0.2, 0.3, 0.5],
    "reg_lambda"       : [0.5, 1.0, 1.5, 2.0],
    "reg_alpha"        : [0, 0.05, 0.1, 0.3, 0.5],
}

# Instancias de SageMaker
TRAINING_INSTANCE  = "ml.m5.xlarge"
ENDPOINT_INSTANCE  = "ml.m5.large"
XGBOOST_VERSION    = "1.7-1"

# LLM de Bedrock
BEDROCK_MODEL_ID   = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# Configuración de download.py
URL = "https://data.insideairbnb.com/mexico/df/mexico-city/2025-09-27/data/listings.csv.gz"
OUTPUT = "airbnb_final.parquet"
COLS = [
    "neighbourhood_cleansed", "latitude", "longitude", "room_type",
    "accommodates", "bathrooms", "bedrooms", "beds", "amenities", "price",
]

PARKING_PATTERN = r"Free parking on premises|Paid parking on premises"
PATIO_PATTERN   = r"Shared patio or balcony|Private patio or balcony|Patio or balcony"

# Outliers a excluir por subconjunto
ENTIRE_BEDROOMS_DROP  = {20, 21, 25, 50}
PRIVATE_BEDROOMS_DROP = {0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 18, 20, 23, 40}
PRIVATE_BEDS_DROP     = {8, 12, 16}
PRIVATE_BATHS_DROP    = {9, 10, 12, 13, 15}
PRIVATE_ACCOM_DROP    = {9, 10, 14, 16}
PRIVATE_MAX_PRICE     = 15_000