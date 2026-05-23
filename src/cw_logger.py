"""
cw_logger.py — Helper de logging hacia CloudWatch + consola
Todos los scripts importan get_logger() de aquí.
Los logs se envían simultáneamente a:
  - Consola (stdout) → visibles en el notebook de Studio
  - AWS CloudWatch Logs → grupo configurado en config.CW_LOG_GROUP
"""

import logging
import time
import json
import boto3
import botocore.exceptions
import config


class CloudWatchHandler(logging.Handler):
    """
    Handler de logging que envía registros a un log stream de CloudWatch.
    Agrupa mensajes en lotes para reducir llamadas a la API.
    """
    # configuración inicial
    def __init__(self, log_group: str, stream_name: str, region: str):
        super().__init__()
        self.log_group   = log_group
        self.stream_name = stream_name
        self.region      = region
        self._client     = boto3.client("logs", region_name=region)
        self._seq_token  = None
        self._buffer     = []
        self._buf_size   = 10        # flush cada 10 mensajes
        self._last_flush = time.time()
        self._flush_interval = 5.0  # o cada 5 segundos

        self._ensure_log_group()
        self._ensure_log_stream()

    # configuración del log group
    def _ensure_log_group(self) -> None:
        try:
            self._client.create_log_group(logGroupName=self.log_group)
        except self._client.exceptions.ResourceAlreadyExistsException:
            pass
        except Exception as e:
            print(f"[CW] No se pudo crear log group: {e}")
    
    # configuración del log stream
    def _ensure_log_stream(self) -> None:
        try:
            self._client.create_log_stream(
                logGroupName  = self.log_group,
                logStreamName = self.stream_name,
            )
        except self._client.exceptions.ResourceAlreadyExistsException:
            # Obtener el sequence token actual
            try:
                resp = self._client.describe_log_streams(
                    logGroupName       = self.log_group,
                    logStreamNamePrefix= self.stream_name,
                    limit              = 1,
                )
                streams = resp.get("logStreams", [])
                if streams:
                    self._seq_token = streams[0].get("uploadSequenceToken")
            except Exception:
                pass
        except Exception as e:
            print(f"[CW] No se pudo crear log stream: {e}")


    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._buffer.append({
                "timestamp": int(record.created * 1000),
                "message"  : msg,
            })
            now = time.time()
            if (len(self._buffer) >= self._buf_size or
                    now - self._last_flush >= self._flush_interval):
                self._flush()
        except Exception:
            self.handleError(record)

    # envío de logs
    def _flush(self) -> None:
        if not self._buffer:
            return
        try:
            kwargs = {
                "logGroupName" : self.log_group,
                "logStreamName": self.stream_name,
                "logEvents"    : sorted(self._buffer, key=lambda e: e["timestamp"]),
            }
            if self._seq_token:
                kwargs["sequenceToken"] = self._seq_token

            resp = self._client.put_log_events(**kwargs)
            self._seq_token = resp.get("nextSequenceToken")
            self._buffer    = []
            self._last_flush = time.time()

        except self._client.exceptions.InvalidSequenceTokenException as e:
            # Recuperar el token correcto del mensaje de error
            import re
            match = re.search(r"sequenceToken is: (\S+)", str(e))
            if match:
                self._seq_token = match.group(1)
                self._flush()   # reintentar con el token correcto
        except Exception as e:
            print(f"[CW] Error al enviar logs: {e}")
            self._buffer = []

    def close(self) -> None:
        self._flush()
        super().close()


# creación de logger que va a estar en cada script
_loggers: dict[str, logging.Logger] = {}

def get_logger(
    name        : str,
    stream_name : str = "general",
    level       : int = logging.INFO,
) -> logging.Logger:
    """
    Devuelve un logger configurado con:
      - StreamHandler → consola (visible en notebook)
      - CloudWatchHandler → CloudWatch Logs
    """
    key = f"{name}:{stream_name}"
    if key in _loggers:
        return _loggers[key]

    logger = logging.getLogger(key)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        fmt     = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )

    # console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # servicio de cloudwatch
    try:
        cw = CloudWatchHandler(
            log_group   = config.CW_LOG_GROUP,
            stream_name = stream_name,
            region      = config.REGION,
        )
        cw.setLevel(level)
        cw.setFormatter(fmt)
        logger.addHandler(cw)
    except Exception as e:
        logger.warning(f"CloudWatch logging no disponible: {e}")

    _loggers[key] = logger
    return logger