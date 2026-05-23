"""
preprocess_airbnb.py
Pipeline de preprocesamiento para datos de Airbnb (Inside Airbnb).
Descarga el CSV comprimido, aplica filtros y transformaciones, y
exporta el resultado como Parquet.
"""

import gzip
import io
import urllib.request
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from cw_logger import get_logger
import config

logger = get_logger(__name__, stream_name="download")

URL    = config.URL
OUTPUT = config.OUTPUT
PARKING_PATTERN = config.PARKING_PATTERN
PATIO_PATTERN   = config.PATIO_PATTERN
ENTIRE_BEDROOMS_DROP  = config.ENTIRE_BEDROOMS_DROP
PRIVATE_BEDROOMS_DROP = config.PRIVATE_BEDROOMS_DROP
PRIVATE_BEDS_DROP     = config.PRIVATE_BEDS_DROP
PRIVATE_BATHS_DROP    = config.PRIVATE_BATHS_DROP
PRIVATE_ACCOM_DROP    = config.PRIVATE_ACCOM_DROP
PRIVATE_MAX_PRICE     = config.PRIVATE_MAX_PRICE
COLS                  = config.COLS

def _read_gz_bytes(data: bytes) -> pd.DataFrame:
    """Descomprime bytes .gz y devuelve un DataFrame."""
    with gzip.open(io.BytesIO(data)) as f:
        return pd.read_csv(f, low_memory=False)

def download(url: str = URL) -> pd.DataFrame:
    """
    Descarga y descomprime el CSV. Soporta tres fuentes:

    - s3://bucket/key    boto3 (recomendado en SageMaker)
    - /ruta o ./ruta     archivo local en disco
    - https://...        HTTP (bloqueado desde IPs de AWS/servidores)
    """
    # Primer caso, lee el dataset de S3(caso preferido)
    if url.startswith("s3://"):
        import boto3
        logger.info(f"Leyendo desde S3: {url}")
        path = url[5:]
        bucket, key = path.split("/", 1)
        s3 = boto3.client("s3")
        buf = io.BytesIO()
        s3.download_fileobj(bucket, key, buf)
        data = _read_gz_bytes(buf.getvalue())
 
    # Segundo caso, el archivo es local
    elif not url.startswith("http"):
        import os
        if not os.path.exists(url):
            raise FileNotFoundError(f"No se encontró el archivo: {url}")
        logger.info(f"Leyendo archivo local: {url}")
        with open(url, "rb") as f:
            data = _read_gz_bytes(f.read())
 
    # Tercer caso, el archivo se descarga desde la página web oficial
    # Usualmente funciona para la primer descarga, las descargas subsecuentes las bloquea
    else:
        import requests
        logger.info(f"Descargando: {url}")
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "es-MX,es;q=0.9,en;q=0.8",
            "Referer": "https://insideairbnb.com/",
        }
        response = requests.get(url, headers=headers, timeout=120)
        if response.status_code == 403:
            raise RuntimeError(
                "HTTP 403: Inside Airbnb bloquea descargas desde IPs de AWS/servidores.\n"
                "Usando el archivo previamente descargado:\n"
                f"  run(url='s3://tu-bucket/listings.csv.gz')"
            )
        response.raise_for_status()
        data = _read_gz_bytes(response.content)
 
    logger.info(f"{len(data):,} filas  |  {len(data.columns)} columnas")
    return data

def select_columns(raw_data: pd.DataFrame, cols: list[str] = COLS) -> pd.DataFrame:
    """Conserva sólo las columnas relevantes."""
    return raw_data[cols]

def filter_room_types(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Mantiene únicamente 'Private room' y 'Entire home/apt'."""
    return raw_data[raw_data["room_type"].isin(["Private room", "Entire home/apt"])]

def drop_nulls(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas con cualquier valor nulo."""
    return raw_data.dropna()

def encode_amenities(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Crea indicadores binarios de parking y patio/balcón a partir de
    la columna 'amenities', luego la elimina.
    """
    raw_data = raw_data.copy()
    raw_data["parking"]      = raw_data["amenities"].str.contains(PARKING_PATTERN, na=False).astype(int)
    raw_data["patio_balcon"] = raw_data["amenities"].str.contains(PATIO_PATTERN,   na=False).astype(int)
    return raw_data.drop(columns=["amenities"])

def clean_price(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Convierte 'price' de string con $ y comas a float."""
    raw_data = raw_data.copy()
    raw_data["price"] = raw_data["price"].astype(str).str.replace(r"[\$,]", "", regex=True).astype(float)
    return raw_data

def build_entire(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Filtra y limpia el subconjunto 'Entire home/apt'."""
    vals = ENTIRE_BEDROOMS_DROP
    result = (
        raw_data[raw_data["room_type"] == "Entire home/apt"]
        .query("bedrooms not in @vals")
        .drop_duplicates()
    )
    logger.info(f"raw_data_entire:  {len(result):,} filas")
    return result

def build_private(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Filtra y limpia el subconjunto 'Private room'."""
    vals_bed   = PRIVATE_BEDROOMS_DROP
    vals_beds  = PRIVATE_BEDS_DROP
    vals_baths = PRIVATE_BATHS_DROP
    vals_accom = PRIVATE_ACCOM_DROP
    max_price  = PRIVATE_MAX_PRICE
    result = (
        raw_data[raw_data["room_type"] == "Private room"]
        .query("bedrooms     not in @vals_bed")
        .query("beds         not in @vals_beds")
        .query("bathrooms    not in @vals_baths")
        .query("accommodates not in @vals_accom")
        .query("price        <= @max_price")
        .drop_duplicates()
    )
    logger.info(f"raw_data_private: {len(result):,} filas")
    return result

def combine(raw_data_entire: pd.DataFrame, raw_data_private: pd.DataFrame) -> pd.DataFrame:
    """Une ambos subconjuntos en un único DataFrame."""
    result = pd.concat([raw_data_entire, raw_data_private], ignore_index=True)
    logger.info(f"raw_data_final:   {len(result):,} filas")
    return result

def save_parquet(raw_data: pd.DataFrame, path: str = OUTPUT) -> None:
    """Exporta el DataFrame a formato Parquet."""
    table = pa.Table.from_pandas(raw_data, preserve_index=False)
    pq.write_table(table, path)
    logger.info(f"Guardado como '{path}'")

def run(url: str = URL, output: str = OUTPUT) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo de extremo a extremo.
    """
    logger.info("Descarga")
    raw = download(url)
    logger.info("Preprocesamiento general")
    data = (
        raw
        .pipe(select_columns)
        .pipe(filter_room_types)
        .pipe(drop_nulls)
        .pipe(encode_amenities)
        .pipe(clean_price)
    )
    data_final = combine(build_entire(data), build_private(data))
    save_parquet(data_final, output)
    return data_final

if __name__ == "__main__":
    raw_data = run()
    logger.info(raw_data.describe())