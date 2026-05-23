"""
agente.py
Este agente lee el mensaje del usuario, lo procesa y decide a quien enviar un correo electrónico.

"""

import boto3
import json
from strands import Agent, tool
from strands.models import BedrockModel
from cw_logger import get_logger

logger = get_logger(__name__, stream_name="agente")

from config import (
    REGION,
    SNS_TOPIC_ARN_NOVATO,
    SNS_TOPIC_ARN_EXPERTO,
    BEDROCK_MODEL_ID,
)


# Definición de cliente SNS
sns_client = boto3.client("sns", region_name=REGION)

# Tools del agente
@tool
def enviar_correo_novato(mensaje: str) -> str:
    """
    Envía un correo a raulpopoca73@gmail.com (prospectos sin propiedad)
    cuando el usuario no tiene inmueble pero quiere comprar/invertir.
    """
    logger.info("Llamada a tool enviar_correo_novato")
    try:
        response = sns_client.publish(
            TopicArn=SNS_TOPIC_ARN_NOVATO,
            Subject="Nuevo prospecto: interesado en comprar",
            Message=(
                "Se recibió un mensaje de alguien interesado en adquirir un inmueble.\n\n"
                f"Detalle:\n{mensaje}\n\n"
                "Por favor, contactar para brindar información inicial."
            ),
        )
        return f"Correo enviado a raulpopoca73@gmail.com. MessageId: {response['MessageId']}"
    except Exception as e:
        return f"Error al enviar correo novato: {e}"

@tool
def enviar_correo_experto(mensaje: str) -> str:
    """
    Envía un correo a dianaba25@gmail.com (clientes con propiedad)
    cuando el usuario ya tiene inmueble y quiere información especializada.
    """
    logger.info("Llamada a tool enviar_correo_experto")
    try:
        response = sns_client.publish(
            TopicArn=SNS_TOPIC_ARN_EXPERTO,
            Subject="Nuevo prospecto: solicita asesoría especializada",
            Message=(
                "Se recibió un mensaje de alguien que ya tiene un inmueble y busca info especializada.\n\n"
                f"Detalle:\n{mensaje}\n\n"
                "Por favor, contactar con un asesor senior."
            ),
        )
        return f"Correo enviado a dianaba25@gmail.com. MessageId: {response['MessageId']}"
    except Exception as e:
        return f"Error al enviar correo experto: {e}"

@tool
def clasificar_intencion(texto: str) -> str:
    """
    Clasifica la intención del usuario en base a su texto.
    Retorna 'NOVATO' si no tiene propiedad y quiere comprar/invertir,
    'EXPERTO' si ya tiene propiedad y quiere info especializada,
    o 'DESCONOCIDO' si no encaja en ninguna categoría.
    """
    logger.info("Llamada a tool clasificar_intencion")
    texto_lower = texto.lower()
    
    palabras_novato  = ["no tengo", "quiero comprar", "quiero invertir", "primera vez",
                        "sin casa", "sin departamento", "interesada en comprar",
                        "interesado en comprar", "primera vivienda"]
    palabras_experto = ["ya tengo", "tengo un departamento", "tengo una casa",
                        "información más especializada", "asesoría especializada",
                        "quiero más información", "experto", "especializada"]
    
    hits_novato  = sum(1 for p in palabras_novato  if p in texto_lower)
    hits_experto = sum(1 for p in palabras_experto if p in texto_lower)
    
    if hits_novato > hits_experto:
        return "NOVATO"
    elif hits_experto > hits_novato:
        return "EXPERTO"
    else:
        return "DESCONOCIDO"

@tool
def ignorar_mensaje() -> str:
    """
    Cuando el mensaje del usuario NO está relacionado con bienes raíces,
    compra, venta, inversión o renta de inmuebles, casas o departamentos.
    No envía ningún correo ni realiza ninguna acción.
    """
    logger.info("Llamada a tool ignorar_mensaje")
    print(">>> TOOL EJECUTADA: ignorar_mensaje")
    return "Mensaje ignorado: no está relacionado con bienes raíces."

# Creación del agente
_model = BedrockModel(
    model_id=BEDROCK_MODEL_ID,
    region_name=REGION,
)

_SYSTEM_PROMPT = """
Eres un asistente especializado en bienes raíces.

REGLAS ESTRICTAS — DEBES SEGUIRLAS SIN EXCEPCIÓN:
1. Primero determina si el mensaje está relacionado con bienes raíces
   (compra, venta, inversión, renta, casas, departamentos, inmuebles, propiedades).
2. Si el mensaje NO está relacionado con bienes raíces → LLAMA ignorar_mensaje
   y responde amablemente: "Solo puedo ayudarte con temas de bienes raíces."
3. Si el usuario no tiene inmueble o quiere comprar/invertir → LLAMA enviar_correo_novato.
4. Si el usuario ya tiene inmueble y quiere info especializada → LLAMA enviar_correo_experto.
5. Si tienes dudas sobre el perfil → LLAMA clasificar_intencion primero.
6. NUNCA simules resultados. Siempre ejecuta la tool correspondiente.
7. Responde siempre en español.
"""
# configuración del agente con el LLM definido een config
agente = Agent(
    model=_model,
    system_prompt=_SYSTEM_PROMPT,
    tools=[clasificar_intencion, enviar_correo_novato, enviar_correo_experto, ignorar_mensaje],
)

def procesar_mensaje(texto_usuario: str) -> str:
    """Procesa el texto del usuario y dispara el correo correspondiente."""
    print(f"Texto recibido: {texto_usuario}")
    respuesta = agente(texto_usuario)
    print(f"\nRespuesta del agente:\n{respuesta}")
    return str(respuesta)