"""
preprocessing.py — Feature Engineering + Random Search + subida a S3
Estrategias para manejar desbalance en neighbourhood_cleansed:
  1. Target Encoding con Smoothing bayesiano — encodings más robustos
     para alcaldías con pocas muestras, acercándolos a la media global
  2. Oversampling de alcaldías minoritarias — replica observaciones de
     alcaldías con menos muestras hasta alcanzar un mínimo configurable

Flujo:
  1. Leer el parquet desde S3
  2. Feature engineering con smoothed target encoding
  3. Oversampling de alcaldías minoritarias
  4. Random Search para encontrar mejores hiperparámetros XGBoost
  5. Guardar encodings y mejores hiperparámetros en S3
  6. Split train/val y subida a S3
  7. Logs en CloudWatch
"""

import io
import json
import time
import boto3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import config
import logging
from cw_logger import get_logger

logger = get_logger(__name__, stream_name="preprocessing")

# Carga de datos desde la capa Gold

def load_raw_data() -> pd.DataFrame:
    logger.info(f"Leyendo dataset desde S3 {config.RAW_S3_URI}")
    raw_data = pd.read_parquet(config.RAW_S3_URI)
    logger.info(f"Dataset cargado: {raw_data.shape[0]:,} filas, {raw_data.shape[1]} columnas")
    return raw_data


# Existe un desbalance de alojamientos por alcaldía. Por lo anterior, se aplica
# un smoothing para ponderar las alcadlías, según sea el caso, con las medias locales
# o las medias globales. Adicionalmente, se crean datos sintéticos de alcaldías con 
# pocos alojamientos, únicamente replicando sus filas.

def smoothed_target_encoding(
    raw_data        : pd.DataFrame,
    col       : str,
    target    : str,
    smoothing : int = None,
) -> dict:
    """
    Target Encoding con suavizado bayesiano (smoothing).

    Para grupos con pocas muestras, el encoding se acerca a la media global.
    Para grupos con muchas muestras, el encoding se acerca a la media local.

    Fórmula:
        encoded = (n * local_mean + smoothing * global_mean) / (n + smoothing)

    Donde:
        n            = número de muestras del grupo
        local_mean   = media del target en el grupo
        global_mean  = media global del target
        smoothing    = parámetro de suavizado (configurable en config.py)
    """
    if smoothing is None:
        smoothing = config.SMOOTHING_FACTOR

    global_mean = raw_data[target].mean()

    stats = raw_data.groupby(col)[target].agg(["mean", "count"]).rename(
        columns={"mean": "local_mean", "count": "n"}
    )

    stats["encoded"] = (
        (stats["n"] * stats["local_mean"] + smoothing * global_mean)
        / (stats["n"] + smoothing)
    )

    logger.info(
        f"Smoothed encoding '{col}' → '{target}' "
        f"(smoothing={smoothing}, grupos={len(stats)})"
    )

    minority = stats[stats["n"] < smoothing * 2]
    for group, row in minority.iterrows():
        pct_local  = row["n"] / (row["n"] + smoothing) * 100
        pct_global = smoothing / (row["n"] + smoothing) * 100
        logger.info(
            f"  '{group}' (n={int(row['n'])}): "
            f"{pct_local:.0f}% local + {pct_global:.0f}% global "
            f"→ {row['encoded']:.4f}"
        )

    return stats["encoded"].to_dict()


def smoothed_median_encoding(
    raw_data        : pd.DataFrame,
    col       : str,
    target    : str,
    smoothing : int = None,
) -> dict:
    """
    Versión del smoothed encoding usando mediana en lugar de media.

    Aproxima la mediana suavizada combinando la mediana local con
    la mediana global ponderadas por el tamaño del grupo.
    """
    if smoothing is None:
        smoothing = config.SMOOTHING_FACTOR

    global_median = raw_data[target].median()

    stats = raw_data.groupby(col)[target].agg(["median", "count"]).rename(
        columns={"median": "local_median", "count": "n"}
    )

    stats["encoded"] = (
        (stats["n"] * stats["local_median"] + smoothing * global_median)
        / (stats["n"] + smoothing)
    )

    logger.info(
        f"Smoothed median encoding '{col}' a '{target}' "
        f"(smoothing={smoothing})"
    )
    return stats["encoded"].to_dict()

def oversample_minority_neighbourhoods(
    raw_data           : pd.DataFrame,
    min_samples  : int  = None,
    random_state : int  = 42,
) -> pd.DataFrame:
    """
    Oversampling con reemplazo de alcaldías con menos de min_samples
    observaciones, replicando sus filas hasta alcanzar el mínimo.
    """
    if min_samples is None:
        min_samples = config.OVERSAMPLE_MIN_SAMPLES

    # Recuperar neighbourhood desde neighbourhood_encoded no es posible,
    # así que necesitamos pasarlo como columna auxiliar temporalmente.
    # Se elimina antes de devolver el DataFrame.
    if "_neighbourhood_tmp" not in raw_data.columns:
        logger.warning(
            "Columna '_neighbourhood_tmp' no encontrada."
            "Oversampling omitido."
        )
        return raw_data

    original_size = len(raw_data)
    parts         = []
    oversampled   = []

    for hood, group in raw_data.groupby("_neighbourhood_tmp"):
        if len(group) < min_samples:
            resampled = group.sample(
                n            = min_samples,
                replace      = True,
                random_state = random_state,
            )
            parts.append(resampled)
            oversampled.append((hood, len(group), min_samples))
            logger.info(
                f"Oversample: '{hood}' {len(group)} a {min_samples} muestras"
            )
        else:
            parts.append(group)

    raw_data_balanced = (
        pd.concat(parts)
          .sample(frac=1, random_state=random_state)
          .reset_index(drop=True)
    )

    # Eliminar columna auxiliar
    raw_data_balanced = raw_data_balanced.drop(columns=["_neighbourhood_tmp"])

    logger.info("Oversampling de alcaldías minoritarias:")
    logger.info(f" Umbral mínimo : {min_samples} muestras")
    logger.info(f"Oversampling completado: {original_size:,} → {len(raw_data_balanced):,} filas")
    return raw_data_balanced


# En esta parte se aplica el feature engineering para mejorar predicciones con XGboost

def feature_engineering(raw_data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Aplica FE completo con smoothed target encoding para neighbourhood.

    Cambios respecto a la versión anterior:
      - neighbourhood_encoded   → smoothed_median_encoding (más robusto)
      - neighborhood_median_price → smoothed con precio MXN
      - Se conserva columna auxiliar '_neighbourhood_tmp' para oversampling
    """
    logger.info("Iniciando feature engineering")
    raw_data = raw_data.copy()

    # Price es el target a predecir entonces se elimina
    raw_data = raw_data[raw_data["price"] > 0].reset_index(drop=True)
    raw_data["log_price"] = np.log1p(raw_data["price"])

    # Columna necesaria para oversampling
    raw_data["_neighbourhood_tmp"] = raw_data["neighbourhood_cleansed"]

    # encoding binario para private room
    raw_data["is_private_room"] = (raw_data["room_type"] == "Private room").astype(int)

    # ── Target encoding room_type (pocas categorías, sin smoothing) ──────────
    room_encoded: dict = (
        raw_data.groupby("room_type")["log_price"].median().to_dict()
    )
    raw_data["room_type_encoded"] = raw_data["room_type"].map(room_encoded)

    room_med_price: dict = (
        raw_data.groupby("room_type")["price"].median().to_dict()
    )
    raw_data["room_type_median_price"] = raw_data["room_type"].map(room_med_price)

    # Se obtienen 2 proporciones(ratios)
    # 1. Número de ocupantes / habitaciones
    # 2. Baños / Número de ocupantes
    raw_data["guests_per_room"] = raw_data["accommodates"] / raw_data["bedrooms"].replace(0, 1)
    raw_data["beds_per_guest"]  = raw_data["beds"]          / raw_data["accommodates"].replace(0, 1)

    # Se suman las 2 amenidades. El maximo valor es 2 y el mínimo es 0.
    raw_data["amenity_score"] = raw_data["parking"] + raw_data["patio_balcon"]

    # Se usa el smoothing encoding con el log price
    neigh_encoded: dict = smoothed_median_encoding(
        raw_data, "neighbourhood_cleansed", "log_price",
        smoothing = config.SMOOTHING_FACTOR,
    )
    raw_data["neighbourhood_encoded"] = raw_data["neighbourhood_cleansed"].map(neigh_encoded)

    # Se usa el smoothing encoding con el price(directo)
    neigh_med_price: dict = smoothed_median_encoding(
        raw_data, "neighbourhood_cleansed", "price",
        smoothing = config.SMOOTHING_FACTOR,
    )
    raw_data["neighborhood_median_price"] = raw_data["neighbourhood_cleansed"].map(neigh_med_price)

    # Se divide el mapa de CDMX en una cuadrícula y asignan a cada propiedad
    # el precio mediano de su zona geográfica. Captura la variación de precio dentro
    # de una misma alcaldía
    lat_bins = pd.qcut(raw_data["latitude"],  q=5, labels=False, duplicates="drop") #franjas de 5 grupos
    lon_bins = pd.qcut(raw_data["longitude"], q=5, labels=False, duplicates="drop")
    raw_data["geo_cluster"] = lat_bins.astype(str) + "_" + lon_bins.astype(str)
    # la frnaja tiene aproximadamente el mismo número de propiedades
    # Se calcula la mediana de log_price por cluster
    geo_encoded: dict = (
        raw_data.groupby("geo_cluster")["log_price"].median().to_dict()
    )
    raw_data["geo_cluster_encoded"] = raw_data["geo_cluster"].map(geo_encoded)

    # Se agrega una interacción de número de ocupantes por baño.
    # Esto se agrega derivado de un análisis bayesiano previo.
    raw_data["capacity_bath"] = raw_data["accommodates"] * raw_data["bathrooms"]

    # Se agrega la distancia que existe respecto al Ángel de la Independencia
    # Que se supone es el punto más turístico de CDMX
    raw_data["dist_centro"] = np.sqrt(
        (raw_data["latitude"]  - config.LAT_CENTER) ** 2 +
        (raw_data["longitude"] - config.LON_CENTER) ** 2
    )

    # Simplemente si hay más de 6 ocupantes se activa una flag
    raw_data["is_large"] = (raw_data["accommodates"] >= 6).astype(int)

    # Se hace la selección final
    cols_final = config.FEATURE_COLS + ["_neighbourhood_tmp"]
    raw_data_final   = raw_data[cols_final].dropna().reset_index(drop=True)

    # encodings es un diccionario que guarda todos los mappings calculados 
    # durante el entrenamiento para poder reproducir exactamente el
    # mismo feature engineering en inferencia, sin acceso al dataset original.
    encodings = {
        "neigh_med_price"      : neigh_med_price,
        "neigh_encoded"        : neigh_encoded,
        "geo_encoded"          : geo_encoded,
        "room_encoded"         : room_encoded,
        "room_med_price"       : room_med_price,
        "smoothing_factor"     : config.SMOOTHING_FACTOR,
        "global_med_price"     : float(raw_data["price"].median()), #fallbacks, esto por si acaso que llegué una nueva alcaldía dentro de CDMX
        "global_neigh_encoded" : float(raw_data["log_price"].median()),
        "global_geo_encoded"   : float(raw_data["log_price"].median()),
        "global_room_encoded"  : float(raw_data["log_price"].median()),
    }

    logger.info(
        f"FE completado: {len(raw_data_final):,} filas, "
        f"{len(config.FEATURE_COLS) - 1} features, "
        f"smoothing={config.SMOOTHING_FACTOR}"
    )
    return raw_data_final, encodings

# Búsqueda de hiperparámetros con Random Search.
def run_random_search(raw_data_eng: pd.DataFrame) -> dict:
    """
    RandomizedSearchCV con XGBoost local.
    Recibe el DataFrame ya balanceado (post-oversampling).
    """
    logger.info(
        f"Iniciando Random Search: {config.RS_N_ITER} iteraciones, "
        f"{config.RS_CV_FOLDS} folds CV"
    )
    t0 = time.time()

    X = raw_data_eng[config.INFERENCE_COLS]
    y = raw_data_eng["log_price"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=config.RS_RANDOM_SEED
    )

    base_model = xgb.XGBRegressor(
        objective    = "reg:squarederror",
        eval_metric  = "rmse",
        random_state = config.RS_RANDOM_SEED,
        verbosity    = 0,
        n_jobs       = -1,
    )

    rs = RandomizedSearchCV(
        estimator           = base_model,
        param_distributions = config.RS_PARAM_DISTRIBUTIONS,
        n_iter              = config.RS_N_ITER,
        cv                  = config.RS_CV_FOLDS,
        scoring             = config.RS_SCORING,
        random_state        = config.RS_RANDOM_SEED,
        n_jobs              = -1,
        verbose             = 1,
        refit               = True,
    )
    rs.fit(X_train, y_train)

    elapsed      = time.time() - t0
    best_params  = rs.best_params_
    best_cv_rmse = -rs.best_score_

    preds   = np.expm1(rs.best_estimator_.predict(X_val))
    actuals = np.expm1(y_val)
    mae     = mean_absolute_error(actuals, preds)
    rmse    = np.sqrt(mean_squared_error(actuals, preds))
    r2      = r2_score(actuals, preds)

    logger.info(f"Random Search completado en {elapsed:.1f}s")
    logger.info(f"Mejores hiperparámetros: {json.dumps(best_params)}")
    logger.info(f"CV RMSE (log): {best_cv_rmse:.4f}")
    logger.info(f"Val MAE : ${mae:,.2f} MXN")
    logger.info(f"Val RMSE: ${rmse:,.2f} MXN")
    logger.info(f"Val R²  : {r2:.4f}")

    results_raw_data = pd.DataFrame(rs.cv_results_)
    top5 = results_raw_data.nsmallest(5, "rank_test_score")[
        ["params", "mean_test_score", "std_test_score"]
    ].copy()
    top5["rmse_cv"] = -top5["mean_test_score"]
    
    return {
        "best_params"  : best_params,
        "val_mae"      : mae,
        "val_rmse"     : rmse,
        "val_r2"       : r2,
        "cv_rmse"      : best_cv_rmse,
        "elapsed_secs" : elapsed,
    }
# Guarda artifactos en S3
def save_artifacts(encodings: dict, rs_results: dict, s3_client) -> None:
    """
    Guarda artifactos en S3, de acuerdo a los nombres definidos en config
    """
    def _put(obj: dict, key: str) -> None:
        s3_client.put_object(
            Bucket = config.BUCKET_NAME,
            Key    = key,
            Body   = json.dumps(obj, ensure_ascii=False, indent=2),
        )
        logger.info(f"Artefacto guardado en s3://{config.BUCKET_NAME}/{key}")

    _put(encodings,  config.ENCODINGS_S3_KEY)
    _put(rs_results, config.RS_RESULTS_S3_KEY)

def split_and_upload(raw_data_balanced: pd.DataFrame, s3_client) -> tuple[str, str]:
    """
    Split 80/20 sobre el DataFrame balanceado.
    Solo sube las columnas de FEATURE_COLS (sin _neighbourhood_tmp).
    """
    # Solo guarda las feature cols
    raw_data_upload = raw_data_balanced[config.FEATURE_COLS]

    train_raw_data, val_raw_data = train_test_split(
        raw_data_upload, test_size=0.2, random_state=42
    )
    logger.info(f"Split: train={len(train_raw_data):,}, val={len(val_raw_data):,}")

    def _upload(raw_data: pd.DataFrame, key: str) -> str:
        buf = io.StringIO()
        raw_data.to_csv(buf, index=False, header=False)
        s3_client.put_object(
            Bucket = config.BUCKET_NAME,
            Key    = key,
            Body   = buf.getvalue(),
        )
        uri = f"s3://{config.BUCKET_NAME}/{key}"
        logger.info(f"Split arriba: {uri}")
        return uri

    train_uri = _upload(train_raw_data, f"{config.PREFIX}/data/train/train.csv")
    val_uri   = _upload(val_raw_data,   f"{config.PREFIX}/data/validation/validation.csv")
    return train_uri, val_uri

# es la función que se ejecuta desde el notebook
def run_preprocessing(
    run_rs     : bool = True,
    oversample : bool = True,
) -> tuple[str, str, dict, dict]:
    """
    Orquesta el pipeline completo.
    """
    logger.info("Inicio del pipeline de preprocesamiento")
    logger.info(
        f"Configuración: smoothing={config.SMOOTHING_FACTOR}, "
        f"oversample_min={config.OVERSAMPLE_MIN_SAMPLES}"
    )

    s3 = boto3.client("s3", region_name=config.REGION)

    raw_data_raw = load_raw_data()
    raw_data_eng, encodings = feature_engineering(raw_data_raw)
    if oversample:
        raw_data_balanced = oversample_minority_neighbourhoods(raw_data_eng)
    else:
        # Eliminala columna temporal si no hace oversampling
        raw_data_balanced = raw_data_eng.drop(columns=["_neighbourhood_tmp"])
        logger.info("Oversampling omitido (oversample=False)")
    # random search
    rs_results = {}
    if run_rs:
        rs_results = run_random_search(raw_data_balanced)
    else:
        logger.info("Random Search omitido (run_rs=False)")

    # almacena artifactos
    logger.info("Guardando artefactos en S3")
    save_artifacts(encodings, rs_results, s3)

    # Splits
    logger.info("Subiendo splits a S3")
    train_uri, val_uri = split_and_upload(raw_data_balanced, s3)

    logger.info("Preprocesamiento completo")
    logger.info(f"   Train : {train_uri}")
    logger.info(f"   Val   : {val_uri}")
    return train_uri, val_uri, encodings, rs_results

if __name__ == "__main__":
    run_preprocessing()