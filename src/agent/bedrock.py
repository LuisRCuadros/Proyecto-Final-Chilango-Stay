"""
bedrock.py
Restricciones implementadas:
  1. Ubicación debe ser CDMX (alcaldía, dirección, colonia, zona)
     Si no es CDMX: mensaje de error con lista de alcaldías
  2. Defaults si no se especifica: 2 ocupantes, 2 recámaras, 2 camas,
     1 baño, sin estacionamiento, sin patio/balcón
  3. beds = bedrooms si no se especifican camas
  4. Dirección/colonia sin alcaldía, geocodificación vía segunda
     llamada a Bedrock (usa conocimiento geográfico del LLM)
  5. Texto sin relación con alojamientos, "Por favor haz una solicitud válida"
  6. Caché de geocodificación persistente en S3: si una dirección ya fue
     geocodificada antes, se reutilizan las coordenadas sin llamar a Bedrock.
"""

import json
import re
import boto3
import numpy as np
import config
from inference import build_features
from cw_logger import get_logger


logger = get_logger(__name__, stream_name="bedrock")


# Constantes
BEDROCK_MODEL_ID  = config.BEDROCK_MODEL_ID
GEO_CACHE_S3_KEY  = f"{config.PREFIX}/artifacts/geo_cache.json"  # caché de geocodificación

# Alcaldías válidas con coordenadas centroide
ALCALDIAS: dict[str, dict] = {
    "Azcapotzalco"          : {"latitude": 19.4747, "longitude": -99.1786},
    "Benito Juárez"         : {"latitude": 19.3840, "longitude": -99.1646},
    "Coyoacán"              : {"latitude": 19.3391, "longitude": -99.1585},
    "Cuajimalpa de Morelos" : {"latitude": 19.3617, "longitude": -99.2798},
    "Cuauhtémoc"            : {"latitude": 19.4216, "longitude": -99.1623},
    "Gustavo A. Madero"     : {"latitude": 19.4808, "longitude": -99.1149},
    "Iztacalco"             : {"latitude": 19.4006, "longitude": -99.0946},
    "Iztapalapa"            : {"latitude": 19.3566, "longitude": -99.0801},
    "La Magdalena Contreras": {"latitude": 19.3162, "longitude": -99.2355},
    "Miguel Hidalgo"        : {"latitude": 19.4320, "longitude": -99.1914},
    "Milpa Alta"            : {"latitude": 19.1981, "longitude": -99.0473},
    "Tlalpan"               : {"latitude": 19.2772, "longitude": -99.1753},
    "Tláhuac"               : {"latitude": 19.2804, "longitude": -99.0189},
    "Venustiano Carranza"   : {"latitude": 19.4259, "longitude": -99.0980},
    "Xochimilco"            : {"latitude": 19.2429, "longitude": -99.1078},
    "Álvaro Obregón"        : {"latitude": 19.3606, "longitude": -99.2134},
}

# Alias de colonias/zonas  alcaldía canónica
ALCALDIA_ALIASES: dict[str, str] = {
    "azcapotzalco"           : "Azcapotzalco",
    "benito juarez"          : "Benito Juárez",
    "benito juárez"          : "Benito Juárez",
    "del valle"              : "Benito Juárez",
    "narvarte"               : "Benito Juárez",
    "napoles"                : "Benito Juárez",
    "nápoles"                : "Benito Juárez",
    "insurgentes"            : "Benito Juárez",
    "coyoacan"               : "Coyoacán",
    "coyoacán"               : "Coyoacán",
    "copilco"                : "Coyoacán",
    "pedregal de san angel"  : "Coyoacán",
    "cuajimalpa"             : "Cuajimalpa de Morelos",
    "cuajimalpa de morelos"  : "Cuajimalpa de Morelos",
    "santa fe"               : "Cuajimalpa de Morelos",
    "cuauhtemoc"             : "Cuauhtémoc",
    "cuauhtémoc"             : "Cuauhtémoc",
    "centro"                 : "Cuauhtémoc",
    "centro historico"       : "Cuauhtémoc",
    "centro histórico"       : "Cuauhtémoc",
    "tepito"                 : "Cuauhtémoc",
    "doctores"               : "Cuauhtémoc",
    "roma"                   : "Cuauhtémoc",
    "roma norte"             : "Cuauhtémoc",
    "roma sur"               : "Cuauhtémoc",
    "condesa"                : "Cuauhtémoc",
    "juarez"                 : "Cuauhtémoc",
    "juárez"                 : "Cuauhtémoc",
    "tabacalera"             : "Cuauhtémoc",
    "gustavo a madero"       : "Gustavo A. Madero",
    "gustavo a. madero"      : "Gustavo A. Madero",
    "gam"                    : "Gustavo A. Madero",
    "lindavista"             : "Gustavo A. Madero",
    "iztacalco"              : "Iztacalco",
    "iztapalapa"             : "Iztapalapa",
    "la magdalena contreras" : "La Magdalena Contreras",
    "magdalena contreras"    : "La Magdalena Contreras",
    "miguel hidalgo"         : "Miguel Hidalgo",
    "polanco"                : "Miguel Hidalgo",
    "lomas de chapultepec"   : "Miguel Hidalgo",
    "chapultepec"            : "Miguel Hidalgo",
    "anahuac"                : "Miguel Hidalgo",
    "anáhuac"                : "Miguel Hidalgo",
    "milpa alta"             : "Milpa Alta",
    "tlalpan"                : "Tlalpan",
    "pedregal"               : "Tlalpan",
    "tlahuac"                : "Tláhuac",
    "tláhuac"                : "Tláhuac",
    "venustiano carranza"    : "Venustiano Carranza",
    "merced"                 : "Venustiano Carranza",
    "xochimilco"             : "Xochimilco",
    "alvaro obregon"         : "Álvaro Obregón",
    "álvaro obregón"         : "Álvaro Obregón",
    "alvaro obregón"         : "Álvaro Obregón",
    "san angel"              : "Álvaro Obregón",
    "san ángel"              : "Álvaro Obregón",
    "mixcoac"                : "Álvaro Obregón",
    "observatorio"           : "Álvaro Obregón",
}

# Valores por default (restricción 2)
DEFAULTS = {
    "room_type"    : "Entire home/apt",
    "accommodates" : 2,
    "bedrooms"     : 2,
    "beds"         : None,
    "bathrooms"    : 1.0,
    "parking"      : 0,
    "patio_balcon" : 0,
}

VALID_ROOM_TYPES = {"Entire home/apt", "Private room"}

# Prompts
# Estos prompts fueron generados con IA.
EXTRACTION_PROMPT = """Eres un asistente especializado en extraer información de alojamientos en Ciudad de México.

Analiza la descripción y extrae estos campos en formato JSON:
- is_valid_listing: true si el texto describe o pregunta sobre un alojamiento (casa, depa, cuarto, hotel, renta, Airbnb, precio de propiedad, hospedaje, etc.), false si no tiene nada que ver con alojamientos.
- location: CUALQUIER referencia geográfica mencionada: alcaldía, colonia, calle, dirección, zona, referencia de metro (string, null si no se menciona nada).
- room_type: "Entire home/apt" si es casa/depa/apartamento/loft/estudio completo, "Private room" si es cuarto/habitación privada dentro de una propiedad compartida. null si no se puede determinar.
- accommodates: número de huéspedes/ocupantes (integer, null si no se menciona)
- bedrooms: número de habitaciones/cuartos/recámaras (integer, null si no se menciona)
- beds: número de camas (integer, null si no se menciona)
- bathrooms: número de baños (float, null si no se menciona)
- parking: tiene estacionamiento — 1=sí, 0=no, null=no mencionado
- patio_balcon: tiene patio o balcón/terraza — 1=sí, 0=no, null=no mencionado

REGLAS:
1. Responde ÚNICAMENTE con el JSON. Sin markdown, sin explicaciones.
2. is_valid_listing=false SOLO si el texto claramente no tiene relación con alojamientos ("hola", "¿qué hora es?", "precio del dólar"). Textos mínimos como "rento cuarto" o "tengo algo en Coyoacán" son valid=true.
3. "recámara" o "cuarto" → bedroom. "cama matrimonial/individual/litera/sofá cama" → bed.
4. "cuarto privado", "habitación privada", "rento mi cuarto" → room_type="Private room".

EJEMPLOS:
{"is_valid_listing": true, "location": "Coyoacán", "room_type": "Entire home/apt", "accommodates": 4, "bedrooms": 2, "beds": null, "bathrooms": 1.0, "parking": 0, "patio_balcon": 1}
{"is_valid_listing": true, "location": "Cerro del Vigilante 91", "room_type": "Private room", "accommodates": 2, "bedrooms": 1, "beds": 1, "bathrooms": 1.0, "parking": null, "patio_balcon": null}
{"is_valid_listing": false, "location": null, "room_type": null, "accommodates": null, "bedrooms": null, "beds": null, "bathrooms": null, "parking": null, "patio_balcon": null}"""

GEOCODING_PROMPT = """Eres un experto en geografía urbana de Ciudad de México (CDMX).

Determina si la ubicación dada pertenece a CDMX e identifica la alcaldía y coordenadas.

ALCALDÍAS VÁLIDAS:
Azcapotzalco, Benito Juárez, Coyoacán, Cuajimalpa de Morelos, Cuauhtémoc,
Gustavo A. Madero, Iztacalco, Iztapalapa, La Magdalena Contreras, Miguel Hidalgo,
Milpa Alta, Tlalpan, Tláhuac, Venustiano Carranza, Xochimilco, Álvaro Obregón

Responde ÚNICAMENTE con JSON:
- Si es CDMX:    {"is_cdmx": true,  "alcaldia": "Nombre exacto", "latitude": 19.XXXX, "longitude": -99.XXXX, "confidence": "high/medium/low"}
- Si NO es CDMX: {"is_cdmx": false, "city_detected": "nombre de la ciudad o lugar detectado"}
- Si no puedes determinar: {"is_cdmx": null, "reason": "explicación breve"}

EJEMPLOS:
"Cerro del Vigilante 91"  → {"is_cdmx": true,  "alcaldia": "Coyoacán",       "latitude": 19.3350, "longitude": -99.1620, "confidence": "high"}
"Col. Del Valle"          → {"is_cdmx": true,  "alcaldia": "Benito Juárez",   "latitude": 19.3840, "longitude": -99.1646, "confidence": "high"}
"metro Polanco"           → {"is_cdmx": true,  "alcaldia": "Miguel Hidalgo",  "latitude": 19.4330, "longitude": -99.1940, "confidence": "high"}
"Guadalajara"             → {"is_cdmx": false, "city_detected": "Guadalajara"}
"calle 5 de mayo"         → {"is_cdmx": null,  "reason": "nombre de calle genérico sin contexto suficiente"}"""

# Llamada a Bedrock
def _invoke_bedrock(system: str, user_msg: str,
                    client, max_tokens: int = 512) -> str:
    """Llamada genérica a Bedrock. Devuelve texto limpio."""
    logger.debug(f"Invocando Bedrock: {BEDROCK_MODEL_ID}")
    response = client.invoke_model(
        modelId     = BEDROCK_MODEL_ID,
        contentType = "application/json",
        accept      = "application/json",
        body        = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens"       : max_tokens,
            "system"           : system,
            "messages"         : [{"role": "user", "content": user_msg}],
        }),
    )
    body = json.loads(response["body"].read())
    raw  = body["content"][0]["text"].strip()
    raw  = re.sub(r"```json\s*", "", raw)
    raw  = re.sub(r"```\s*",     "", raw)
    return raw

# hace el json para el modelo
def _parse_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Bedrock no devolvió JSON válido:\n{raw}")
        
# extrae información del texto del usuario
def extract_fields(user_text: str, client) -> dict:
    """Paso 1: extrae todos los campos del texto libre."""
    raw = _invoke_bedrock(EXTRACTION_PROMPT, user_text, client)
    result = _parse_json(raw)
    logger.info(f"Extracción Bedrock: {json.dumps(result, ensure_ascii=False)}")
    return result

# caché de geocodificación
# Clase GeoCache generada por IA
class GeoCache:
    """
    Caché persistente de geocodificación guardado en S3 como JSON.

    Flujo:
      1. Al inicializar → carga el caché desde S3 en memoria
      2. Antes de llamar a Bedrock → busca en el caché por clave normalizada
      3. Si hay hit → devuelve coordenadas sin llamar a Bedrock (caché HIT)
      4. Si no hay hit → geocodifica con Bedrock y guarda el resultado en S3

    Formato del JSON en S3:
    {
      "amberes 33": {
        "neighbourhood": "Cuauhtémoc",
        "latitude": 19.4320,
        "longitude": -99.1611,
        "location_raw": "Amberes 33",
        "created_at": "2025-01-15T10:30:00"
      },
      ...
    }
    """

    def __init__(self, region: str):
        self._s3     = boto3.client("s3", region_name=region)
        self._bucket = config.BUCKET_NAME
        self._key    = GEO_CACHE_S3_KEY
        self._cache  = self._load()

    def _load(self) -> dict:
        """Carga el caché desde S3. Si no existe, devuelve dict vacío."""
        try:
            obj  = self._s3.get_object(Bucket=self._bucket, Key=self._key)
            data = json.loads(obj["Body"].read().decode("utf-8"))
            logger.info(f"Caché de geocodificación cargado: {len(data)} entradas")
            return data
        except self._s3.exceptions.NoSuchKey:
            logger.info("Caché de geocodificación vacío (primera vez)")
            return {}
        except Exception as e:
            logger.warning(f"No se pudo cargar el caché de geocodificación: {e}")
            return {}

    def _save(self) -> None:
        """Persiste el caché completo en S3."""
        try:
            self._s3.put_object(
                Bucket = self._bucket,
                Key    = self._key,
                Body   = json.dumps(self._cache, ensure_ascii=False, indent=2),
            )
            logger.info(
                f"Caché guardado en s3://{self._bucket}/{self._key} "
                f"({len(self._cache)} entradas)"
            )
        except Exception as e:
            logger.warning(f"No se pudo guardar el caché: {e}")

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """
        Divide el texto en tokens significativos para búsqueda difusa.
        Elimina stopwords y palabras cortas irrelevantes.
        Ej: "calle amberes número 33 col juárez" → {"amberes", "33", "juarez"}
        """
        STOPWORDS = {
            "calle", "avenida", "av", "blvd", "boulevard", "calzada",
            "calz", "cerrada", "privada", "andador", "paseo", "periferico",
            "numero", "num", "no", "col", "colonia", "barrio", "fracc",
            "fraccionamiento", "delegacion", "alcaldia", "entre", "esquina",
            "esq", "y", "de", "del", "la", "el", "los", "las", "con", "piso",
            "depto", "departamento", "edificio", "torre", "interior", "int",
        }
        tokens = set(re.sub(r"[^a-z0-9áéíóúüñ ]", " ", _normalize(text)).split())
        return tokens - STOPWORDS - {""}

    def _fuzzy_match(self, key: str) -> tuple[str, float] | None:
        """
        Búsqueda difusa en tres niveles:
          1. Coincidencia exacta de clave normalizada
          2. Todos los tokens de la consulta están en la clave cacheada
          3. Todos los tokens de la clave cacheada están en la consulta
        Devuelve (cache_key, score) del mejor match o None.
        """
        # Nivel 1: exacto
        if key in self._cache:
            return key, 1.0

        query_tokens = self._tokenize(key)
        if not query_tokens:
            return None

        best_key   = None
        best_score = 0.0

        for cache_key in self._cache:
            cached_tokens = self._tokenize(cache_key)
            if not cached_tokens:
                continue

            # Tokens en común
            common = query_tokens & cached_tokens
            if not common:
                continue

            # Score: Jaccard + bonus si todos los tokens cacheados están presentes
            jaccard = len(common) / len(query_tokens | cached_tokens)

            # Bonus si la consulta contiene todos los tokens de la entrada cacheada
            # (el usuario puede escribir más contexto que lo que está guardado)
            coverage = len(cached_tokens & query_tokens) / len(cached_tokens)
            score    = jaccard * 0.5 + coverage * 0.5

            if score > best_score:
                best_score = score
                best_key   = cache_key

        # Umbral mínimo: al menos 0.6 para evitar falsos positivos
        if best_key and best_score >= 0.6:
            return best_key, best_score
        return None

    def get(self, location_raw: str) -> dict | None:
        """
        Busca la ubicación en el caché con matching difuso.
        Funciona para variaciones de escritura de la misma dirección:
          "Amberes 33"           → guarda como "amberes 33"
          "amberes numero 33"    → match con "amberes 33"  ✓
          "calle amberes 33"     → match con "amberes 33"  ✓
          "av. amsterdam 33"     → NO match con "amberes 33" ✗
        """
        key    = _normalize(location_raw)
        match  = self._fuzzy_match(key)

        if match is None:
            return None

        cache_key, score = match
        entry = self._cache[cache_key]

        logger.info(
            f"Caché HIT para '{location_raw}' "
            f"(match='{entry['location_raw']}', score={score:.2f}): "
            f"{entry['neighbourhood']} "
            f"(lat={entry['latitude']:.4f}, lon={entry['longitude']:.4f})"
        )
        label = "exacto" if score == 1.0 else f"similar ({score:.0%})"
        print(f"Caché HIT [{label}]: '{location_raw}' → "
              f"'{entry['location_raw']}' → "
              f"{entry['neighbourhood']} "
              f"(lat={entry['latitude']:.4f}, lon={entry['longitude']:.4f})")
        return {
            "neighbourhood": entry["neighbourhood"],
            "latitude"     : entry["latitude"],
            "longitude"    : entry["longitude"],
        }

    def put(self, location_raw: str, geo: dict) -> None:
        """
        Guarda una nueva entrada en el caché y persiste en S3.
        geo debe tener: neighbourhood, latitude, longitude.
        """
        from datetime import datetime
        key = _normalize(location_raw)
        self._cache[key] = {
            "neighbourhood": geo["neighbourhood"],
            "latitude"     : geo["latitude"],
            "longitude"    : geo["longitude"],
            "location_raw" : location_raw,
            "created_at"   : datetime.utcnow().isoformat(),
        }
        logger.info(
            f"Caché guardado para '{location_raw}': "
            f"{geo['neighbourhood']} "
            f"(lat={geo['latitude']:.4f}, lon={geo['longitude']:.4f})"
        )
        self._save()

    def list_entries(self) -> None:
        """Muestra todas las entradas del caché."""
        if not self._cache:
            print("El caché de geocodificación está vacío.")
            return
        print(f"\n Caché de geocodificación ({len(self._cache)} entradas):")
        print("─" * 65)
        for key, entry in sorted(self._cache.items()):
            print(
                f"  {entry['location_raw']:<30} → "
                f"{entry['neighbourhood']:<22} "
                f"({entry['latitude']:.4f}, {entry['longitude']:.4f})"
            )
        print("─" * 65)

    def delete(self, location_raw: str) -> bool:
        """Elimina una entrada del caché."""
        key = _normalize(location_raw)
        if key in self._cache:
            del self._cache[key]
            self._save()
            logger.info(f"Entrada eliminada del caché: '{location_raw}'")
            print(f"'{location_raw}' eliminado del caché.")
            return True
        print(f"  '{location_raw}' no encontrado en el caché.")
        return False


def geocode_location(location_text: str, client) -> dict:
    """geocodifica una dirección/colonia con Bedrock."""
    logger.info(f"Geocodificando con Bedrock: '{location_text}'")
    raw = _invoke_bedrock(GEOCODING_PROMPT, location_text, client)
    return _parse_json(raw)


# Resuelve ubicación
def _normalize(text: str) -> str:
    return (text.lower().strip()
            .replace("á","a").replace("é","e")
            .replace("í","i").replace("ó","o")
            .replace("ú","u").replace("ü","u"))


def _closest_alcaldia(lat: float, lon: float) -> str:
    return min(
        ALCALDIAS,
        key=lambda a: (ALCALDIAS[a]["latitude"]  - lat) ** 2
                    + (ALCALDIAS[a]["longitude"] - lon) ** 2,
    )


def _error_no_cdmx(city: str) -> str:
    lista = "\n".join(f"  • {a}" for a in sorted(ALCALDIAS))
    return (
        f"La ubicación '{city}' no está en Ciudad de México.\n"
        f"Por favor proporciona una alcaldía, colonia, dirección o zona de CDMX.\n"
        f"Alcaldías disponibles:\n{lista}"
    )


def resolve_location(
    location_raw : str | None,
    client,
    geo_cache    : "GeoCache | None" = None,
) -> dict | None | str:
    """
    Resuelve ubicación en cascada:
      1. Alias exacto / búsqueda parcial en diccionario, inmediato
      2. Caché de geocodificación en S3, sin llamar a Bedrock
      3. Geocodificación con Bedrock, segunda llamada al LLM,
      si es CDMX: guarda en caché para futuras consultas
      4. Fuera de CDMX, str con mensaje de error
      5. Indeterminable, None (se usarán defaults)
    """
    if not location_raw:
        return None

    key = _normalize(location_raw)

    # Nivel 1: alias exacto
    if key in ALCALDIA_ALIASES:
        alcaldia = ALCALDIA_ALIASES[key]
        return {"neighbourhood": alcaldia, **ALCALDIAS[alcaldia]}

    # Nivel 1b: búsqueda parcial
    for alias, alcaldia in ALCALDIA_ALIASES.items():
        if alias in key or key in alias:
            return {"neighbourhood": alcaldia, **ALCALDIAS[alcaldia]}

    # Nivel 1c: nombre canónico directo
    for alcaldia in ALCALDIAS:
        if _normalize(alcaldia) in key or key in _normalize(alcaldia):
            return {"neighbourhood": alcaldia, **ALCALDIAS[alcaldia]}

    # Nivel 2: caché de geocodificación
    if geo_cache is not None:
        cached = geo_cache.get(location_raw)
        if cached is not None:
            return cached

    # Nivel 3: geocodificación con Bedrock
    print(f" Geocodificando '{location_raw}' con Bedrock")
    try:
        geo = geocode_location(location_raw, client)
    except Exception as e:
        logger.warning(f"Error al geocodificar '{location_raw}': {e}")
        return None

    if geo.get("is_cdmx") is True:
        alcaldia = geo.get("alcaldia", "")
        lat      = geo.get("latitude")
        lon      = geo.get("longitude")
        conf     = geo.get("confidence", "low")

        if alcaldia not in ALCALDIAS and lat and lon:
            alcaldia = _closest_alcaldia(lat, lon)

        if alcaldia in ALCALDIAS:
            if conf == "low" or not lat or not lon:
                lat = ALCALDIAS[alcaldia]["latitude"]
                lon = ALCALDIAS[alcaldia]["longitude"]
            logger.info(
                f"Geocodificado: {alcaldia} "
                f"(lat={lat:.4f}, lon={lon:.4f}, conf={conf})"
            )
            print(f"Alcaldía: {alcaldia} "
                  f"(lat={lat:.4f}, lon={lon:.4f}, conf={conf})")
            result = {"neighbourhood": alcaldia, "latitude": lat, "longitude": lon}
            # Guardar en caché para futuras consultas (restricción 6)
            if geo_cache is not None:
                geo_cache.put(location_raw, result)
            return result
        return None

    elif geo.get("is_cdmx") is False:
        city = geo.get("city_detected", location_raw)
        logger.warning(f"Ubicación fuera de CDMX: {city}")
        return _error_no_cdmx(city)

    else:
        reason = geo.get("reason", "desconocido")
        logger.warning(f"Ubicación indeterminable: {reason}")
        print(f"Ubicación no determinada: {reason}")
        return None


# Defaults y validación
def apply_defaults(extracted: dict, geo: dict | None) -> dict:
    """Aplica defaults para campos ausentes y resuelve coordenadas."""

    # Room type
    room_type = extracted.get("room_type")
    if room_type not in VALID_ROOM_TYPES:
        room_type = DEFAULTS["room_type"]

    accommodates = extracted.get("accommodates") or DEFAULTS["accommodates"]
    bedrooms     = extracted.get("bedrooms")     or DEFAULTS["bedrooms"]
    bathrooms    = extracted.get("bathrooms")    or DEFAULTS["bathrooms"]
    parking      = extracted.get("parking")
    patio_balcon = extracted.get("patio_balcon")
    if parking      is None: parking      = DEFAULTS["parking"]
    if patio_balcon is None: patio_balcon = DEFAULTS["patio_balcon"]

    # Restricción 3: beds = bedrooms si no se especificó
    beds = extracted.get("beds") or int(bedrooms)

    # Coordenadas
    if geo:
        neighbourhood = geo["neighbourhood"]
        latitude      = geo["latitude"]
        longitude     = geo["longitude"]
    else:
        neighbourhood = "Cuauhtémoc"
        latitude      = 19.4216
        longitude     = -99.1623

    fields = {
        "neighbourhood" : neighbourhood,
        "room_type"     : room_type,
        "latitude"      : latitude,
        "longitude"     : longitude,
        "accommodates"  : int(accommodates),
        "bedrooms"      : int(bedrooms),
        "beds"          : int(beds),
        "bathrooms"     : float(bathrooms),
        "parking"       : int(parking),
        "patio_balcon"  : int(patio_balcon),
    }
    logger.info(f"Campos resueltos: {json.dumps(fields, ensure_ascii=False)}")
    return fields


# Definición de pipeline principal
class AirbnbBedrockPipeline:
    """
    Pipeline completo: texto libre -> Bedrock -> validación -> SageMaker -> precio.

    Ejemplos
    --------
    pipeline = AirbnbBedrockPipeline()

    # Alcaldía explícita
    pipeline.predict_from_text(
        "Depa completo en Coyoacán, 2 recámaras, balcón, sin estacionamiento"
    )
    # Dirección
    pipeline.predict_from_text(
        "Rento mi cuarto en Cerro del Vigilante 91, para 2 personas"
    )
    # Colonia
    pipeline.predict_from_text(
        "Departamento en la Condesa, 1 baño, estudio pequeño con terraza"
    )
    # Sin relación, mensaje de error
    pipeline.predict_from_text("¿Cuánto cuesta el dólar hoy?")
    # Fuera de CDMX, genera mensaje de error
    pipeline.predict_from_text("Casa en Guadalajara, 3 recámaras")
    """

    def __init__(
        self,
        endpoint_name : str = config.ENDPOINT_NAME,
        region        : str = config.REGION,
        bedrock_region: str = None,
    ):
        self.endpoint_name  = endpoint_name
        self.region         = region
        self.bedrock_region = bedrock_region or region

        self._bedrock = boto3.client(
            "bedrock-runtime", region_name=self.bedrock_region
        )
        self._sm_runtime = boto3.client(
            "sagemaker-runtime", region_name=self.region
        )
        self._encodings = self._load_encodings()
        self._geo_cache  = GeoCache(region=self.region)

        logger.info(
            f"Pipeline inicializado | endpoint={endpoint_name} | "
            f"model={BEDROCK_MODEL_ID}"
        )
        print(f"Pipeline listo.")
        print(f"Bedrock model : {BEDROCK_MODEL_ID}")
        print(f"SM endpoint   : {endpoint_name}")

    def _load_encodings(self) -> dict:
        s3  = boto3.client("s3", region_name=self.region)
        obj = s3.get_object(
            Bucket=config.BUCKET_NAME, Key=config.ENCODINGS_S3_KEY
        )
        return json.loads(obj["Body"].read().decode("utf-8"))

    def _call_endpoint(self, csv_payload: str) -> float:
        resp = self._sm_runtime.invoke_endpoint(
            EndpointName = self.endpoint_name,
            ContentType  = "text/csv",
            Body         = csv_payload,
        )
        raw = resp["Body"].read().decode().strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                log_price = float(parsed["predictions"][0]["score"])
            elif isinstance(parsed, list):
                log_price = float(parsed[0])
            else:
                log_price = float(parsed)
        except (json.JSONDecodeError, KeyError, IndexError):
            log_price = float(raw)
        return round(np.expm1(log_price), 2)

    def _print_table(self, fields: dict, price: float,
                     geo_source: str = "") -> None:
        print("\n" + "─" * 54)
        print("Información extraída del alojamiento")
        print("─" * 54)
        rows = [
            ("Alcaldía",
             f"{fields['neighbourhood']}"
             + (f"  [{geo_source}]" if geo_source else "")),
            ("Tipo de alojamiento", fields["room_type"]),
            ("Latitud",            f"{fields['latitude']:.4f}"),
            ("Longitud",           f"{fields['longitude']:.4f}"),
            ("Ocupantes",          fields["accommodates"]),
            ("Habitaciones",       fields["bedrooms"]),
            ("Camas",              fields["beds"]),
            ("Baños",              fields["bathrooms"]),
            ("Estacionamiento",    "✓ Sí" if fields["parking"]      else "✗ No"),
            ("Patio/Balcón",       "✓ Sí" if fields["patio_balcon"] else "✗ No"),
        ]
        for label, value in rows:
            print(f"  {label:<22}: {value}")
        print("─" * 54)
        print(f"Precio estimado : ${price:>12,.2f} MXN / noche")
        print("─" * 54 + "\n")

    def predict_from_text(self, user_text: str) -> dict | str:
        """
        Procesa descripción en texto libre y devuelve precio estimado.

        Returns
        -------
        dict : {"fields": {...}, "price": float}
        str  : mensaje de error
        """
        logger.info(f"Solicitud recibida: '{user_text[:100]}'")

        # Paso 1: extracción de campos
        print("[1/3] Extrayendo campos con Bedrock")
        extracted = extract_fields(user_text, self._bedrock)
        print(f"   JSON extraído: {json.dumps(extracted, ensure_ascii=False)}")

        # Restricción 5: solicitud válida
        if not extracted.get("is_valid_listing", True):
            msg = "Por favor haz una solicitud válida relacionada con un alojamiento."
            logger.warning(f"Solicitud inválida: '{user_text[:80]}'")
            print(msg)
            return msg

        # Paso 2: resolución de ubicación
        print("[2/3] Resolviendo ubicación")
        location_raw = extracted.get("location")
        geo_source   = ""

        geo = resolve_location(location_raw, self._bedrock, self._geo_cache)

        # Restricción 1: fuera de CDMX
        if isinstance(geo, str):
            print(geo)
            return geo

        if geo is None:
            geo_source = "default (Cuauhtémoc)" if location_raw else ""
        else:
            key = _normalize(location_raw or "")
            if key in ALCALDIA_ALIASES or any(
                _normalize(a) in key for a in ALCALDIAS
            ):
                geo_source = "alias"
            else:
                geo_source = "geocodificado por Bedrock"

        # Paso 3: defaults + FE + inferencia
        print("[3/3] Calculando precio con SageMaker")
        fields      = apply_defaults(extracted, geo)
        csv_payload = build_features(
            neighbourhood = fields["neighbourhood"],
            room_type     = fields["room_type"],
            latitude      = fields["latitude"],
            longitude     = fields["longitude"],
            accommodates  = fields["accommodates"],
            bathrooms     = fields["bathrooms"],
            bedrooms      = fields["bedrooms"],
            beds          = fields["beds"],
            parking       = fields["parking"],
            patio_balcon  = fields["patio_balcon"],
            encodings     = self._encodings,
        )
        price = self._call_endpoint(csv_payload)

        logger.info(
            f"Predicción exitosa: ${price:,.2f} MXN | "
            f"neighbourhood={fields['neighbourhood']} | "
            f"room_type={fields['room_type']}"
        )
        self._print_table(fields, price, geo_source)
        return {"fields": fields, "price": price}

    def run(self) -> None:
        """Modo interactivo en consola/terminal."""
        print("\n" + "=" * 54)
        print("Airbnb Price Predictor v2 — CDMX")
        print("=" * 54)
        print("Describe tu alojamiento en lenguaje natural.")
        print("Escribe 'salir' para terminar.\n")

        while True:
            try:
                user_input = input("Tu alojamiento: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n ¡Hasta luego!")
                break
            if not user_input:
                continue
            if user_input.lower() in ("salir", "exit", "quit"):
                print("¡Hasta luego!")
                break
            self.predict_from_text(user_input)

if __name__ == "__main__":
    pipeline = AirbnbBedrockPipeline()
    pipeline.run()