import json
import os
from pathlib import Path
from typing import Dict
from io import BytesIO
from datetime import datetime

import boto3
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from botocore.exceptions import ClientError

# Módulos del pipeline desarrollados para este proyecto
import config
from bedrock import ALCALDIAS, AirbnbBedrockPipeline
from inference import build_features, load_encodings
from agente import procesar_mensaje


# Configuración de propiedades generales de Streamlit
# layout="wide" permite usar mejor el ancho de pantalla en gráficas y tablas
st.set_page_config(
    page_title="Estimador de precio Airbnb CDMX",
    layout="wide",
)

# Configuración externa para facilitar despliegue en AWS
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME", config.ENDPOINT_NAME)
REGION = os.getenv("AWS_REGION", config.REGION)

# Ubicación del archivo en Amazon S3
S3_BUCKET = os.getenv("S3_BUCKET", "airbnb-pred")
S3_DATA_KEY = os.getenv("S3_DATA_KEY", "airbnb-price/raw/airbnb_final.parquet")

# Carpeta de S3 donde se guardan los registros de contacto de usuarios
S3_OUTPUT_PREFIX = os.getenv("S3_OUTPUT_PREFIX", "airbnb-price/output")

# Rangos de los controles del formulario
# La tupla se interpreta como: mínimo, máximo, valor inicial y paso
# El valor inicial establecido es la mediana de cada variable
SLIDER_HUESPEDES = (1, 16, 4, 1)
SLIDER_RECAMARAS = (0, 6, 2, 1)
SLIDER_CAMAS = (0, 9, 2, 1)
SLIDER_BANOS = (0.5, 5.0, 1.0, 0.5)

# Tipos de alojamiento que se muestran en la interfaz
ROOM_TYPES = ["Entire home/apt", "Private room"]

# Columnas mínimas que se necesitan del archivo para construir
# referencias de mercado por alcaldía
COLUMNAS_REQUERIDAS = ["neighbourhood_cleansed", "price"]


# Recursos cacheados
@st.cache_resource
def get_sagemaker_runtime():
    """Crear un cliente de SageMaker Runtime.

    Este cliente se usa para invocar el servicio de estimación de precios.
    Se cachea como recurso para no recrearlo en cada interacción del usuario.

    Returns:
        Cliente boto3 configurado para SageMaker Runtime.
    """
    return boto3.client("sagemaker-runtime", region_name=REGION)


@st.cache_resource
def get_sagemaker_client():
    """Crear un cliente de SageMaker.

    Este cliente se usa únicamente para revisar si el servicio de estimación
    está disponible antes de habilitar los botones de estimación.

    Returns:
        Cliente boto3 configurado para SageMaker.
    """
    return boto3.client("sagemaker", region_name=REGION)


@st.cache_resource
def get_encodings() -> dict:
    """Cargar los encodings usados por el pipeline de inferencia.

    Los encodings permiten transformar variables categóricas, como alcaldía
    y tipo de alojamiento, al formato esperado por el servicio de estimación.

    Returns:
        Diccionario con los encodings del modelo.
    """
    return load_encodings()


@st.cache_resource
def get_bedrock_pipeline():
    """Inicializar el pipeline que interpreta descripciones en texto libre.

    Este recurso se usa en la pestaña "Describir alojamiento" para extraer
    características del inmueble a partir de una descripción escrita por el
    usuario.

    Returns:
        Instancia de AirbnbBedrockPipeline.
    """
    return AirbnbBedrockPipeline(endpoint_name=ENDPOINT_NAME, region=REGION)


@st.cache_data(ttl=300)
def endpoint_status() -> str:
    """Consultar el estado del servicio de estimación.

    Returns:
        Estado del endpoint. Por ejemplo: InService, Creating, Failed,
        NotFound o Unavailable.
    """
    try:
        response = get_sagemaker_client().describe_endpoint(EndpointName=ENDPOINT_NAME)
        return response["EndpointStatus"]
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")
        if error_code in {"ValidationException", "ResourceNotFound"}:
            return "NotFound"
        return "Unavailable"


@st.cache_data(ttl=3600)  # Cache por 1 hora para no releer S3 en cada rerun
def load_market_data() -> pd.DataFrame:
    """Cargar el archivo procesado de Airbnb desde Amazon S3.

    El archivo vive en un bucket de S3, de modo que la app 
    puede leerlo sin necesidad de incluirlo en la imagen Docker. La
    ubicación se controla con variables de entorno para facilitar el
    despliegue en distintos ambientes.

    Returns:
        DataFrame con el archivo procesado de alojamientos.

    Raises:
        ValueError: si el archivo no contiene las columnas requeridas.
    """
    # Lee el objeto parquet directamente desde S3 con boto3
    s3 = boto3.client("s3", region_name=REGION)
    response = s3.get_object(Bucket=S3_BUCKET, Key=S3_DATA_KEY)
    df = pd.read_parquet(BytesIO(response["Body"].read())).copy()

    # Valida que existan las columnas necesarias para las referencias por alcaldía
    missing_cols = [col for col in COLUMNAS_REQUERIDAS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas requeridas en el archivo: {missing_cols}")

    # Homologa la columna de precio a formato numérico
    if df["price"].dtype == "object":
        df["price"] = (
            df["price"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )

    # Convierte a numérico. Los valores no convertibles se vuelven NaN
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Conserva únicamente precios válidos y positivos
    df = df[df["price"].notna() & (df["price"] > 0)].copy()

    # Asegura que la alcaldía quede como texto para agrupar correctamente
    df["neighbourhood_cleansed"] = df["neighbourhood_cleansed"].astype(str)

    return df


@st.cache_data(ttl=3600)  # Cache por 1 hora para agilizar la navegación de la app
def get_precios_por_alcaldia() -> pd.DataFrame:
    """Calcular referencias de mercado por alcaldía.

    Las referencias se calculan directamente desde airbnb_final.parquet.
    No se dejan precios escritos manualmente en el código, de modo que la
    pestaña de mercado siempre sea trazable al archivo del proyecto.

    Returns:
        DataFrame con precio típico, precio promedio y número de alojamientos
        observados por alcaldía.
    """
    # Carga el archivo de alojamientos
    df = load_market_data()

    # Agrupa por alcaldía y calcula referencias de mercado
    resumen = (
        df.groupby("neighbourhood_cleansed", as_index=False)
        .agg(
            precio_tipico=("price", "median"),
            precio_promedio=("price", "mean"),
            alojamientos_observados=("price", "count"),
        )
        .sort_values("precio_tipico", ascending=False)
    )

    # Redondea precios para presentarlos de forma más limpia al usuario
    resumen["precio_tipico"] = resumen["precio_tipico"].round(0)
    resumen["precio_promedio"] = resumen["precio_promedio"].round(0)

    return resumen


def formato_mxn(valor: float) -> str:
    """Formatear un valor numérico a pesos.

    Args:
        valor: Número que se desea mostrar como precio.

    Returns:
        Texto en formato '$X,XXX MXN'. Si el valor no existe, devuelve 'N/D'.
    """
    if pd.isna(valor):
        return "N/D"
    return f"${valor:,.0f} MXN"


def invoke_endpoint(csv_payload: str) -> float:
    """Invocar el servicio de estimación y transformar la respuesta a pesos.

    El servicio devuelve la predicción en escala logarítmica, por lo que se
    transforma de regreso con expm1 para obtener el precio aproximado por noche.

    Args:
        csv_payload: Variables del alojamiento en formato CSV, en el orden
        esperado por el servicio de estimación.

    Returns:
        Precio estimado por noche en MXN.
    """
    # Envía las variables al servicio de estimación
    response = get_sagemaker_runtime().invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="text/csv",
        Body=csv_payload,
    )

    # Lee la respuesta del servicio
    raw = response["Body"].read().decode("utf-8").strip()

    # La respuesta puede venir como JSON o como texto plano, dependiendo de la
    # configuración del contenedor de inferencia. Se manejan ambos casos
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            log_price = float(parsed["predictions"][0]["score"])
        elif isinstance(parsed, list):
            log_price = float(parsed[0])
        else:
            log_price = float(parsed)
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        log_price = float(raw)

    # Regresa el precio a escala original
    return round(float(np.expm1(log_price)), 2)


def estimar_precio(campos: Dict) -> float:
    """Estimar el precio sugerido por noche para un alojamiento.

    Args:
        campos: Diccionario con características del alojamiento, incluyendo
        alcaldía, tipo de alojamiento, huéspedes, recámaras, camas, baños y
        amenidades principales.

    Returns:
        Precio estimado por noche en MXN.
    """
    # Construye las variables finales del modelo con la función del pipeline
    payload = build_features(encodings=get_encodings(), **campos)

    # Invoca el endpoint y regresa el precio estimado
    return invoke_endpoint(payload)


def rango_precio(precio: float) -> pd.DataFrame:
    """Construir un rango de publicación para el inversionista.

    Args:
        precio: Precio base sugerido por la app.

    Returns:
        DataFrame con tres estrategias: conservadora, recomendada y agresiva.
    """
    # Construye tres escenarios alrededor del precio base
    df = pd.DataFrame(
        {
            "Estrategia": ["Conservador", "Recomendado", "Agresivo"],
            "Precio sugerido": [precio * 0.90, precio, precio * 1.10],
            "Cuándo usarlo": [
                "Para atraer primeras reservas o competir en una zona con mucha oferta",
                "Como precio inicial de publicación",
                "Si el alojamiento tiene ubicación, calidad o amenidades superiores",
            ],
        }
    )

    # Da formato de moneda a la columna de precio para mostrarla en la tabla
    df["Precio sugerido"] = df["Precio sugerido"].apply(formato_mxn)

    return df

def construir_campos_desde_formulario(
    alcaldia: str,
    room_type: str,
    accommodates: int,
    bedrooms: int,
    beds: int,
    bathrooms: float,
    parking: bool,
    patio_balcon: bool,
) -> Dict:
    """Convertir los inputs del formulario al formato del pipeline.

    Args:
        alcaldia: Alcaldía seleccionada por el usuario.
        room_type: Tipo de alojamiento.
        accommodates: Número de huéspedes que admite el alojamiento.
        bedrooms: Número de recámaras.
        beds: Número de camas.
        bathrooms: Número de baños.
        parking: Indicador de estacionamiento.
        patio_balcon: Indicador de patio o balcón.

    Returns:
        Diccionario con las variables esperadas por la función build_features.
    """
    return {
        "neighbourhood": alcaldia,
        "room_type": room_type,
        "latitude": ALCALDIAS[alcaldia]["latitude"],
        "longitude": ALCALDIAS[alcaldia]["longitude"],
        "accommodates": int(accommodates),
        "bathrooms": float(bathrooms),
        "bedrooms": int(bedrooms),
        "beds": int(beds),
        "parking": int(parking),
        "patio_balcon": int(patio_balcon),
    }


def obtener_referencia_alcaldia(alcaldia: str) -> Dict:
    """Obtener referencia de mercado para una alcaldía.

    Args:
        alcaldia: Alcaldía para la cual se quiere consultar el precio típico.

    Returns:
        Diccionario con precio típico y número de alojamientos observados.
    """
    # Obtiene la tabla agregada por alcaldía
    df_ref = get_precios_por_alcaldia()

    # Filtra la alcaldía seleccionada
    fila = df_ref[df_ref["neighbourhood_cleansed"] == alcaldia]
    if fila.empty:
        return {"precio_tipico": np.nan, "alojamientos_observados": 0}

    # Extrae los valores relevantes
    row = fila.iloc[0]
    return {
        "precio_tipico": row["precio_tipico"],
        "alojamientos_observados": int(row["alojamientos_observados"]),
    }


def mostrar_resultado_precio(precio: float, campos: Dict):
    """Mostrar el resultado de estimación de forma amigable.

    Args:
        precio: Precio sugerido por noche.
        campos: Características del alojamiento usadas en la estimación.
    """
    # Extrae variables principales para construir la explicación
    alcaldia = campos["neighbourhood"]
    room_type = campos["room_type"]

    # Obtiene la referencia de mercado para la alcaldía seleccionada
    referencia = obtener_referencia_alcaldia(alcaldia)
    precio_tipico = referencia["precio_tipico"]
    alojamientos_observados = referencia["alojamientos_observados"]

    st.write("### Precio sugerido de publicación")

    # Métricas principales para el inversionista
    col1, col2, col3 = st.columns(3)
    col1.metric("Precio sugerido para este alojamiento", formato_mxn(precio))
    col2.metric("Precio típico de alojamientos en esta zona", formato_mxn(precio_tipico))
    col3.metric("Alojamientos observados en la zona", f"{alojamientos_observados:,.0f}")

    # Texto corto para explicar la referencia sin saturar la interfaz

    # Tabla con rangos sugeridos de publicación.
    st.write("### Rango sugerido de publicación")
    st.caption(
        "El rango se construye alrededor del precio sugerido, con una variación de 10% "
        "para representar estrategias de publicación conservadora y agresiva."
    )
    df_rango = rango_precio(precio)
    st.dataframe(df_rango, use_container_width=True, hide_index=True)

    # Explicación de negocio para interpretar el resultado
    st.write("### Lectura para el inversionista")
    mensaje = (
        f"Para un alojamiento tipo **{room_type}** en **{alcaldia}**, "
        f"con capacidad para **{campos['accommodates']} huésped(es)**, "
        f"**{campos['bedrooms']} recámara(s)**, **{campos['beds']} cama(s)** "
        f"y **{campos['bathrooms']} baño(s)**, el precio inicial sugerido es "
        f"de aproximadamente **{formato_mxn(precio)} por noche**."
    )

    # Agrega una interpretación comparando contra la referencia de mercado
    if not pd.isna(precio_tipico):
        if precio > precio_tipico * 1.15:
            mensaje += (
                " Este valor se encuentra por encima del precio típico de la zona, "
                "lo que puede justificarse si el alojamiento tiene buena ubicación, "
                "amenidades atractivas o condiciones superiores frente a otros inmuebles comparables."
            )
        elif precio < precio_tipico * 0.85:
            mensaje += (
                " Este valor se encuentra por debajo del precio típico de la zona, "
                "por lo que podría existir margen para ajustar el precio si el alojamiento "
                "cuenta con atributos competitivos."
            )
        else:
            mensaje += (
                " Este valor se encuentra cercano al precio típico de la zona, "
                "por lo que parece alineado con el mercado observado."
            )

    st.info(mensaje)

    # Expander con el detalle de inputs. Se mantiene oculto para no saturar la vista
    with st.expander("Ver características consideradas"):
        legible = {
            "Alcaldía": campos["neighbourhood"],
            "Tipo de alojamiento": campos["room_type"],
            "Huéspedes": campos["accommodates"],
            "Recámaras": campos["bedrooms"],
            "Camas": campos["beds"],
            "Baños": campos["bathrooms"],
            "Estacionamiento": "Sí" if campos["parking"] else "No",
            "Patio o balcón": "Sí" if campos["patio_balcon"] else "No",
        }
        st.table(pd.Series(legible, name="Valor").to_frame())


def manejar_error_inferencia(error: Exception):
    """Mostrar un error controlado cuando no se puede estimar el precio.

    Args:
        error: Excepción capturada durante la estimación.
    """
    if isinstance(error, ClientError):
        st.error(
            "El servicio de estimación no está disponible en este momento. "
            "Intenta nuevamente más tarde."
        )
    else:
        st.error(f"No fue posible generar la estimación: {error}")


def guardar_contacto_en_s3(nombre: str, correo: str, descripcion: str) -> None:
    """Guardar un registro de contacto del usuario en Amazon S3.

    Cada solicitud se guarda como un archivo JSON independiente dentro de la
    carpeta de salida del bucket, usando la fecha y hora como nombre para
    evitar que un registro sobrescriba a otro.

    Args:
        nombre: Nombre que escribió el usuario
        correo: Correo electrónico del usuario
        descripcion: Texto con la situación que describió
    """
    # Construye el registro como diccionario
    registro = {
        "nombre": nombre.strip(),
        "correo": correo.strip(),
        "descripcion": descripcion.strip(),
        "fecha_registro": datetime.utcnow().isoformat() + "Z",
    }

    # Nombre del archivo (usa la marca de tiempo para que sea un registro único)
    marca = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    key = f"{S3_OUTPUT_PREFIX}/contacto_{marca}.json"

    # Sube el registro a S3 como JSON
    s3 = boto3.client("s3", region_name=REGION)
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=json.dumps(registro, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )


# Título, subtítulo y logo de la app
logo_path = Path(__file__).resolve().parent / "logo" / "logo.svg"

col_title, col_logo = st.columns([3, 1])
with col_title:
    st.title("¿Cuánto cobrar por tu alojamiento en CDMX?")
    st.caption(
        "Estima un precio inicial de publicación por noche con base en las "
        "características de tu alojamiento"
    )
with col_logo:
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)
        
# Carga de las referencias de mercado. Si falla, se detiene la app porque la
# primera pestaña depende de esta información
try:
    df_mercado = get_precios_por_alcaldia()
except Exception as exc:  # noqa: BLE001
    st.error(f"No se pudieron cargar los datos de mercado: {exc}")
    st.stop()

# Revisa si el servicio de estimación está disponible
estado = endpoint_status()
servicio_estimacion_disponible = estado == "InService"

# Mensaje funcional para el usuario. No menciona detalles técnicos del endpoint
if not servicio_estimacion_disponible:
    st.warning(
        "El servicio de estimación no está disponible en este momento. "
        "Puedes consultar el contexto de mercado por alcaldía mientras se restablece."
    )

# Las pestañas se ordenan de lo general a lo particular:
# primero contexto de mercado, luego estimación individual y escenarios.
tab_mercado, tab_texto, tab_form, tab_sensibilidad, tab_contacto = st.tabs(
    [
        "Precios por alcaldía",
        "Describir alojamiento",
        "Estimar precio",
        "Análisis de sensibilidad",
        "Contacto",
    ]
)

# 1. Pestaña "Precios por alcaldía"
with tab_mercado:
    st.write(
        "Consulta el precio típico por noche observado en cada alcaldía de Ciudad de México."
    )

    # Ordena la tabla de mercado de mayor a menor precio típico
    df_top = df_mercado.sort_values("precio_tipico", ascending=False).copy()

    # Identifica las alcaldías con mayor y menor precio típico
    top = df_top.iloc[0]
    bottom = df_top.iloc[-1]

    # Tarjetas principales
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Zona con precio típico más alto",
        top["neighbourhood_cleansed"],
        delta=None,
    )
    c1.caption(f"{formato_mxn(top['precio_tipico'])} por noche")

    c2.metric(
        "Zona con precio típico más bajo",
        bottom["neighbourhood_cleansed"],
        delta=None,
    )
    c2.caption(f"{formato_mxn(bottom['precio_tipico'])} por noche")

    c3.write("Rango de precios típicos")
    c3.markdown(
        f"<div style='font-size: 1.9rem; font-weight: 400;'>"
        f"{formato_mxn(df_top['precio_tipico'].min())} - {formato_mxn(df_top['precio_tipico'].max())}"
        f"</div>",
        unsafe_allow_html=True,
    )
    c3.caption("Precios típicos por noche")


    # Gráfica de barras por alcaldía
    fig = px.bar(
        df_top,
        x="precio_tipico",
        y="neighbourhood_cleansed",
        orientation="h",
        text="precio_tipico",
        title="Precio típico por noche por alcaldía",
        template="plotly_white",        
        hover_data={
            "precio_tipico": ":,.0f",
            "precio_promedio": ":,.0f",
            "alojamientos_observados": ":,.0f",
            "neighbourhood_cleansed": False,
        },
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        xaxis_title="Precio típico por noche",
        yaxis_title="",
    )
    fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    # Tabla de referencia para usuarios que quieran consultar el detalle
    st.write("### Detalle por alcaldía")
    tabla_mercado = df_top.rename(
        columns={
            "neighbourhood_cleansed": "Alcaldía",
            "precio_tipico": "Precio típico por noche",
            "precio_promedio": "Precio promedio por noche",
            "alojamientos_observados": "Alojamientos observados",
        }
    )

    # Da formato de precio a las columnas monetarias
    tabla_mercado["Precio típico por noche"] = tabla_mercado["Precio típico por noche"].apply(formato_mxn)
    tabla_mercado["Precio promedio por noche"] = tabla_mercado["Precio promedio por noche"].apply(formato_mxn)
    st.dataframe(tabla_mercado, use_container_width=True, hide_index=True)

# 2. Pestaña "Describir alojamiento"
with tab_texto:
    st.subheader("Describe el alojamiento")
    st.write(
        "También puedes describir el inmueble en texto libre. La app identificará las "
        "características principales y generará una estimación de precio."
    )

    # Caja de texto para que el usuario describa el alojamiento sin llenar el formulario
    texto = st.text_area(
        "Descripción del alojamiento",
        placeholder=(
            "Ejemplo: Tengo un departamento en la Condesa para 4 personas, "
            "con 2 recámaras, 1 baño, balcón y sin estacionamiento."
        ),
        height=140,
    )

    # Botón para interpretar la descripción y estimar precio
    if st.button("Estimar desde la descripción", type="primary", disabled=not servicio_estimacion_disponible):
        if not texto.strip():
            st.info("Escribe primero una descripción del alojamiento.")
        else:
            try:
                # El pipeline interpreta el texto y después estima el precio
                with st.spinner("Interpretando descripción y calculando precio sugerido..."):
                    pipeline = get_bedrock_pipeline()
                    resultado = pipeline.predict_from_text(texto)

                # Si el pipeline regresa texto, se muestra como advertencia
                if isinstance(resultado, str):
                    st.warning(resultado)
                else:
                    campos = resultado["fields"]
                    precio = resultado["price"]
                    st.session_state["ultima_estimacion"] = {"campos": campos, "precio": precio}
                    mostrar_resultado_precio(precio, campos)

                    st.caption(
                        "Si alguna característica no coincide con el inmueble, puedes especificar mejor "
                        "los datos en el recuadro de texto libre de esta pestaña o ajustar los datos "
                        "manualmente en la pestaña Estimar precio."
                    )

            except Exception as exc: 
                manejar_error_inferencia(exc)

# 3. Pestaña "Estimar precio"
with tab_form:
    st.subheader("Características del alojamiento")
    st.write(
        "Completa la información del alojamiento para obtener un precio inicial sugerido "
        "de publicación."
    )

    # Se divide el formulario en dos columnas para aprovechar mejor el espacio
    col_izq, col_der = st.columns(2)

    with col_izq:
        # Variables de ubicación y tipo de inmueble
        alcaldia = st.selectbox("Alcaldía", sorted(ALCALDIAS.keys()))
        room_type = st.selectbox("Tipo de alojamiento", ROOM_TYPES)

        # Variables de capacidad.
        accommodates = st.slider(
            "Huéspedes que admite",
            SLIDER_HUESPEDES[0],
            SLIDER_HUESPEDES[1],
            SLIDER_HUESPEDES[2],
            step=SLIDER_HUESPEDES[3],
        )
        bedrooms = st.slider(
            "Recámaras",
            SLIDER_RECAMARAS[0],
            SLIDER_RECAMARAS[1],
            SLIDER_RECAMARAS[2],
            step=SLIDER_RECAMARAS[3],
        )

    with col_der:
        # Variables físicas del alojamiento
        beds = st.slider(
            "Camas",
            SLIDER_CAMAS[0],
            SLIDER_CAMAS[1],
            SLIDER_CAMAS[2],
            step=SLIDER_CAMAS[3],
        )
        bathrooms = st.slider(
            "Baños",
            SLIDER_BANOS[0],
            SLIDER_BANOS[1],
            SLIDER_BANOS[2],
            step=SLIDER_BANOS[3],
        )

        # Amenidades principales
        parking = st.toggle("Tiene estacionamiento")
        patio_balcon = st.toggle("Tiene patio o balcón")

    # Botón de estimación. Se deshabilita si el servicio no está disponible
    if st.button("Estimar precio", type="primary", disabled=not servicio_estimacion_disponible):
        # Construye el diccionario de variables a partir del formulario
        campos = construir_campos_desde_formulario(
            alcaldia=alcaldia,
            room_type=room_type,
            accommodates=accommodates,
            bedrooms=bedrooms,
            beds=beds,
            bathrooms=bathrooms,
            parking=parking,
            patio_balcon=patio_balcon,
        )

        try:
            # Calcula el precio sugerido y lo muestra al usuario
            with st.spinner("Calculando precio sugerido..."):
                precio = estimar_precio(campos)
            st.session_state["ultima_estimacion"] = {"campos": campos, "precio": precio}
            mostrar_resultado_precio(precio, campos)
        except Exception as exc:  # noqa: BLE001
            manejar_error_inferencia(exc)


# 4. Pestaña "Análisis de sensibilidad"
with tab_sensibilidad:
    st.subheader("Análisis de sensibilidad")
    st.write(
        "Este análisis ayuda a visualizar cómo podría cambiar el precio sugerido al modificar "
        "una característica del alojamiento, manteniendo las demás condiciones iguales."
    )

    st.markdown("**1. Define el alojamiento base**")

    # Formulario base para definir el alojamiento que se quiere analizar
    col_a, col_b = st.columns(2)

    with col_a:
        s_alcaldia = st.selectbox("Alcaldía", sorted(ALCALDIAS.keys()), key="sens_alcaldia")
        s_room = st.selectbox("Tipo de alojamiento", ROOM_TYPES, key="sens_room")
        s_acc = st.slider(
            "Huéspedes",
            SLIDER_HUESPEDES[0],
            SLIDER_HUESPEDES[1],
            SLIDER_HUESPEDES[2],
            step=SLIDER_HUESPEDES[3],
            key="sens_acc",
        )
        s_bed = st.slider(
            "Recámaras",
            SLIDER_RECAMARAS[0],
            SLIDER_RECAMARAS[1],
            SLIDER_RECAMARAS[2],
            step=SLIDER_RECAMARAS[3],
            key="sens_bed",
        )

    with col_b:
        s_beds = st.slider(
            "Camas",
            SLIDER_CAMAS[0],
            SLIDER_CAMAS[1],
            SLIDER_CAMAS[2],
            step=SLIDER_CAMAS[3],
            key="sens_beds",
        )
        s_bath = st.slider(
            "Baños",
            SLIDER_BANOS[0],
            SLIDER_BANOS[1],
            SLIDER_BANOS[2],
            step=SLIDER_BANOS[3],
            key="sens_bath",
        )
        s_parking = st.toggle("Estacionamiento", key="sens_parking")
        s_patio = st.toggle("Patio o balcón", key="sens_patio")

    st.markdown("**2. Elige la característica a modificar**")

    # Variable que se modificará en el análisis de sensibilidad
    caracteristica = st.selectbox(
        "Característica",
        ["Huéspedes", "Recámaras", "Camas", "Baños", "Estacionamiento", "Patio o balcón"],
    )

    # Rango de valores que se evaluarán para cada característica
    rangos_sensibilidad = {
        "Huéspedes": ("accommodates", list(range(1, 17))),
        "Recámaras": ("bedrooms", list(range(0, 7))),
        "Camas": ("beds", list(range(0, 10))),
        "Baños": ("bathrooms", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
        "Estacionamiento": ("parking", [0, 1]),
        "Patio o balcón": ("patio_balcon", [0, 1]),
    }

    # Ejecuta el análisis de sensibilidad.
    if st.button("Analizar sensibilidad", type="primary", disabled=not servicio_estimacion_disponible):
        # Construye el alojamiento base.
        base = construir_campos_desde_formulario(
            alcaldia=s_alcaldia,
            room_type=s_room,
            accommodates=s_acc,
            bedrooms=s_bed,
            beds=s_beds,
            bathrooms=s_bath,
            parking=s_parking,
            patio_balcon=s_patio,
        )

        # Identifica la variable interna y los valores que se evaluarán
        campo, valores = rangos_sensibilidad[caracteristica]

        try:
            resultados = []
            progreso = st.progress(0.0, text="Calculando escenarios...")

            # Para cada valor, modifica una sola característica y conserva iguales
            # las demás condiciones del alojamiento
            for i, valor in enumerate(valores):
                escenario = dict(base)
                escenario[campo] = valor
                precio = estimar_precio(escenario)
                resultados.append(
                    {
                        caracteristica: valor,
                        "Precio estimado": precio,
                    }
                )
                progreso.progress((i + 1) / len(valores), text="Calculando escenarios...")

            progreso.empty()
            df_sens = pd.DataFrame(resultados)

            st.write("### Resultados")

            # Para variables binarias, se muestra una gráfica de barras
            if campo in {"parking", "patio_balcon"}:
                df_sens[caracteristica] = df_sens[caracteristica].map({0: "Sin", 1: "Con"})

                fig_binaria = px.bar(
                    df_sens,
                    x=caracteristica,
                    y="Precio estimado",
                    text="Precio estimado",
                    title=f"Cambio en el precio sugerido según {caracteristica.lower()}",
                    template="plotly_white",
                )
                fig_binaria.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
                fig_binaria.update_layout(yaxis_title="Precio sugerido por noche", xaxis_title="")
                st.plotly_chart(fig_binaria, use_container_width=True)

                # Tabla con formato de precio para lectura de negocio
                tabla_sens = df_sens.copy()
                tabla_sens["Precio estimado"] = tabla_sens["Precio estimado"].apply(formato_mxn)
                st.dataframe(tabla_sens, use_container_width=True, hide_index=True)

            # Para variables numéricas, se muestra una línea de sensibilidad
            else:
                fig_sens = px.line(
                    df_sens,
                    x=caracteristica,
                    y="Precio estimado",
                    markers=True,
                    title=f"Cambio en el precio sugerido según {caracteristica.lower()}",
                    template="plotly_white",
                )
                fig_sens.update_layout(
                    xaxis_title=caracteristica,
                    yaxis_title="Precio sugerido por noche",
                )
                st.plotly_chart(fig_sens, use_container_width=True)

                # Tabla con formato de precio para lectura de negocio
                tabla_sens = df_sens.copy()
                tabla_sens["Precio estimado"] = tabla_sens["Precio estimado"].apply(formato_mxn)
                st.dataframe(tabla_sens, use_container_width=True, hide_index=True)

        except Exception as exc: 
            manejar_error_inferencia(exc)

# 5. Pestaña "Contacto"

with tab_contacto:
    st.subheader("¿Quieres asesoría?")
    st.write(
        "Déjanos tus datos y cuéntanos tu situación. Si ya tienes un inmueble "
        "y quieres rentarlo, o si estás pensando en comprar para invertir, "
        "te canalizaremos con la persona del equipo que pueda ayudarte."
    )

    # Aviso de privacidad
    st.caption(
        "Al enviar este formulario, tu nombre y correo se almacenarán de forma "
        "privada, para que el equipo pueda contactarte. No se compartirán tus datos con "
        "terceros."
    )

    # Datos de contacto del usuario
    col_nombre, col_correo = st.columns(2)
    with col_nombre:
        nombre_contacto = st.text_input("Tu nombre", key="nombre_contacto")
    with col_correo:
        correo_contacto = st.text_input("Tu correo electrónico", key="correo_contacto")

    # Descripción de la situación 
    texto_contacto = st.text_area(
        "Describe tu situación",
        placeholder=(
            "Ejemplo: Ya tengo un departamento en la Narvarte y quiero "
            "empezar a rentarlo en Airbnb."
        ),
        height=140,
        key="texto_contacto",
    )

    if st.button("Enviar mi solicitud", type="primary"):
        # Validación mínima
        if not nombre_contacto.strip():
            st.info("Escribe tu nombre.")
        elif not correo_contacto.strip() or "@" not in correo_contacto:
            st.info("Escribe un correo electrónico válido.")
        elif not texto_contacto.strip():
            st.info("Escribe una breve descripción de tu situación.")
        else:
            try:
                with st.spinner("Procesando tu solicitud..."):
                    # 1. El agente clasifica la intención y envía el correo
                    #    al miembro del equipo correspondiente.
                    respuesta = procesar_mensaje(texto_contacto)

                    # 2. Se guarda el registro de contacto en S3
                    guardar_contacto_en_s3(
                        nombre=nombre_contacto,
                        correo=correo_contacto,
                        descripcion=texto_contacto,
                    )

                st.success("Tu solicitud fue procesada. Gracias por contactarnos.")
                st.write(respuesta)
            except Exception as exc: 
                st.error(
                    f"No fue posible procesar tu solicitud en este momento: {exc}"
                )