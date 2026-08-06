import logging


def get_pylogger(name: str = __name__) -> logging.Logger:
    """CosyVoice 추론용 — lightning 없이 표준 logging만 사용."""
    return logging.getLogger(name)
