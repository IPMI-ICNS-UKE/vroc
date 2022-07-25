import logging


class LoggerMixin:
    @property
    def logger(self):
        try:
            self.__logger
        except AttributeError:
            logger_name = f"{self.__module__}.{self.__class__.__name__}"
            self.__logger = logging.getLogger(logger_name)
        return self.__logger


class LogFormatter(logging.Formatter):
    FORMAT = (
        "{asctime} [{levelname:>8}] [{name:>32}:{funcName:>32}:L{lineno:>4}] {message}"
    )

    def __init__(self, datefmt=None, validate=True):
        super().__init__(fmt=self.FORMAT, datefmt=datefmt, style="{", validate=validate)

    def _shorten_left(self, s: str, max_length: int = 32) -> str:
        if len(s) > max_length:
            s = "..." + s[-(max_length - 3) :]

        return s

    def formatMessage(self, record: logging.LogRecord) -> str:

        record.name = self._shorten_left(record.name, max_length=32)
        record.funcName = self._shorten_left(record.funcName, max_length=32)

        return super().formatMessage(record)
