"""
inference.py — Inferencia en tiempo real (SageMaker SDK 2.257.3)
"""

import json
import boto3
import numpy as np
import pandas as pd
import config
from cw_logger import get_logger


logger = get_logger(__name__, stream_name="inference")


# Carga encodings
def load_encodings() -> dict:
    s3  = boto3.client("s3", region_name=config.REGION)
    obj = s3.get_object(Bucket=config.BUCKET_NAME, Key=config.ENCODINGS_S3_KEY)
    enc = json.loads(obj["Body"].read().decode("utf-8"))
    logger.info(f"Encodings cargados desde s3://{config.BUCKET_NAME}/{config.ENCODINGS_S3_KEY}")
    return enc

# construye las features para inferir
def build_features(
    neighbourhood : str,
    room_type     : str,
    latitude      : float,
    longitude     : float,
    accommodates  : int,
    bathrooms     : float,
    bedrooms      : int,
    beds          : int,
    parking       : int,
    patio_balcon  : int,
    encodings     : dict,
) -> str:
    """
    Reproduce el FE de preprocessing.py sobre una observación nueva.
    Devuelve CSV string listo para el endpoint (sin log_price).
    """
    enc = encodings

    # Encodings de categorías
    room_type_encoded     = enc["room_encoded"].get(
        room_type, enc["global_room_encoded"])
    room_type_median_price = enc["room_med_price"].get(
        room_type, enc["global_med_price"])
    neighbourhood_encoded = enc["neigh_encoded"].get(
        neighbourhood, enc["global_neigh_encoded"])
    neighborhood_median_price = enc["neigh_med_price"].get(
        neighbourhood, enc["global_med_price"])
    geo_cluster_encoded   = enc["geo_encoded"].get(
        neighbourhood, enc["global_geo_encoded"])

    # Features derivadas
    guests_per_room  = accommodates / max(bedrooms, 1)
    beds_per_guest   = beds / max(accommodates, 1)
    amenity_score    = parking + patio_balcon
    capacity_bath    = accommodates * bathrooms
    dist_centro      = np.sqrt(
        (latitude  - config.LAT_CENTER) ** 2 +
        (longitude - config.LON_CENTER) ** 2
    )
    is_large         = int(accommodates >= 6)
    is_private_room  = int(room_type == "Private room")

    # Orden exacto = INFERENCE_COLS
    values = [
        latitude, longitude,
        accommodates, bathrooms, bedrooms, beds,
        parking, patio_balcon,
        room_type_encoded,
        guests_per_room, beds_per_guest, amenity_score,
        neighborhood_median_price, neighbourhood_encoded,
        room_type_median_price,
        geo_cluster_encoded, capacity_bath,
        dist_centro, is_large, is_private_room,
    ]
    assert len(values) == len(config.INFERENCE_COLS), (
        f"Mismatch: {len(values)} valores vs "
        f"{len(config.INFERENCE_COLS)} columnas esperadas."
    )
    return ",".join(map(str, values))