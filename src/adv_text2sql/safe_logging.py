"""Redact database credentials before records reach console, file or parent handlers."""

import logging
import os
import re
from urllib.parse import quote, quote_plus

from sqlalchemy.engine import URL, make_url


_passwords: set[str] = set()


def register_database_url(uri) -> None:
    password = make_url(uri).password
    if password:
        _passwords.add(str(password))


def database_label(uri) -> str:
    """Only host/port/database; omit credentials and connection query parameters."""
    url = make_url(uri)
    return URL.create(url.drivername, host=url.host, port=url.port,
                      database=url.database).render_as_string()


def redact_secrets(value) -> str:
    result = str(value)
    secrets = _passwords | {os.getenv("DB_PASS", ""), os.getenv("LLM_API_KEY", "")}
    representations = {v for secret in secrets if secret
                       for v in (secret, quote(secret, safe=""), quote_plus(secret))}
    for secret in sorted(representations, key=len, reverse=True):
        result = result.replace(secret, "***")
    # Also mask credentials in previously unseen URIs, including exception text.
    return re.sub(r"([a-z][a-z0-9+.-]*://)[^\s'\"<>]*@", r"\1***@", result,
                  flags=re.IGNORECASE)


class SecretRedactionFilter(logging.Filter):
    def filter(self, record):
        record.msg = redact_secrets(record.getMessage())
        record.args = ()
        if record.exc_info:
            record.exc_text = redact_secrets(logging.Formatter().formatException(record.exc_info))
            record.exc_info = None
        elif record.exc_text:
            record.exc_text = redact_secrets(record.exc_text)
        if record.stack_info:
            record.stack_info = redact_secrets(record.stack_info)
        return True


def protect_logger(logger) -> None:
    # Handler filters also cover records propagated by descendant loggers.
    for target in (logger, *logger.handlers):
        if not any(isinstance(f, SecretRedactionFilter) for f in target.filters):
            target.addFilter(SecretRedactionFilter())
