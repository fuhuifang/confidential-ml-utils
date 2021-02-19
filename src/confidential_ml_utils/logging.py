# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Utilities around logging data which may or may not contain private content.
"""


from argparse import Namespace
from confidential_ml_utils.constants import DataCategory
import logging
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL
import sys
from threading import Lock
from typing import Any, Dict,Optional, Union
import warnings


_LOCK = Lock()
_PREFIX = None


def set_prefix(prefix: str) -> None:
    """
    Set the global prefix to use when logging public (non-private) data.

    This method is thread-safe.
    """
    with _LOCK:
        global _PREFIX
        _PREFIX = prefix


def get_prefix() -> Optional[str]:
    """
    Obtain the current global prefix to use when logging public (non-private)
    data.
    """
    return _PREFIX


class ConfidentialLogger(logging.getLoggerClass()):  # type: ignore
    """
    Subclass of the default logging class with an explicit `category` parameter
    on all logging methods. It will pass an `extra` param with `prefix` key
    (value depending on whether `category` is public or private) to the
    handlers.

    The default value for data `category` is `PRIVATE` for all methods.

    Implementation is inspired by:
    https://github.com/python/cpython/blob/3.8/Lib/logging/__init__.py
    """

    def __init__(self, name: str):
        super().__init__(name)  # type: ignore

    def _log(self, level, msg, category, args, **kwargs):
        p = ""
        if category == DataCategory.PUBLIC:
            p = get_prefix()
        super(ConfidentialLogger, self)._log(
            level, msg, args, extra={"prefix": p}, **kwargs
        )

    def debug(
        self, msg: str, category: DataCategory = DataCategory.PRIVATE, *args, **kwargs
    ):
        """
        Log `msg % args` with severity `DEBUG`.

        To log public (non-private) data, use the keyword argument `category`
        with a `DataCategory.PUBLIC` value, e.g.

            logger.debug("public data", category=DataCategory.PUBLIC)
        """
        if self.isEnabledFor(DEBUG):
            self._log(DEBUG, msg, category, args, **kwargs)

    def info(
        self, msg: str, category: DataCategory = DataCategory.PRIVATE, *args, **kwargs
    ):
        """
        Log `msg % args` with severity `INFO`.

        To log public (non-private) data, use the keyword argument `category`
        with a `DataCategory.PUBLIC` value, e.g.

            logger.info("public data", category=DataCategory.PUBLIC)
        """
        if self.isEnabledFor(INFO):
            self._log(INFO, msg, category, args, **kwargs)

    def warning(
        self, msg: str, category: DataCategory = DataCategory.PRIVATE, *args, **kwargs
    ):
        """
        Log `msg % args` with severity `WARNING`.

        To log public (non-private) data, use the keyword argument `category`
        with a `DataCategory.PUBLIC` value, e.g.

            logger.warning("public data", category=DataCategory.PUBLIC)
        """
        if self.isEnabledFor(WARNING):
            self._log(WARNING, msg, category, args, **kwargs)

    def warn(
        self, msg: str, category: DataCategory = DataCategory.PRIVATE, *args, **kwargs
    ):
        warnings.warn(
            "The 'warn' method is deprecated, ue 'warning' instead",
            DeprecationWarning,
            2,
        )
        self.warning(msg, category, *args, **kwargs)

    def error(
        self, msg: str, category: DataCategory = DataCategory.PRIVATE, *args, **kwargs
    ):
        """
        Log `msg % args` with severity `ERROR`.

        To log public (non-private) data, use the keyword argument `category`
        with a `DataCategory.PUBLIC` value, e.g.

            logger.error("public data", category=DataCategory.PUBLIC)
        """
        if self.isEnabledFor(ERROR):
            self._log(ERROR, msg, category, args, **kwargs)

    def critical(
        self, msg: str, category: DataCategory = DataCategory.PRIVATE, *args, **kwargs
    ):
        """
        Log `msg % args` with severity `CRITICAL`.

        To log public (non-private) data, use the keyword argument `category`
        with a `DataCategory.PUBLIC` value, e.g.

            logger.critical("public data", category=DataCategory.PUBLIC)
        """
        if self.isEnabledFor(CRITICAL):
            self._log(CRITICAL, msg, category, args, **kwargs)


_logging_basic_config_set_warning = """
********************************************************************************
The root logger already has handlers set! As a result, the behavior of this
library is undefined. If running in Python >= 3.8, this library will attempt to
call logging.basicConfig(force=True), which will remove all existing root
handlers. See https://stackoverflow.com/q/20240464 and
https://github.com/Azure/confidential-ml-utils/issues/33 for more information.
********************************************************************************
"""


def enable_confidential_logging(prefix: str = "SystemLog:", **kwargs) -> None:
    """
    The default format is `logging.BASIC_FORMAT` (`%(levelname)s:%(name)s:%(message)s`).
    All other kwargs are passed to `logging.basicConfig`. Sets the default
    logger class and root logger to be confidential. This means the format
    string `%(prefix)` will work.

    Set the format using the `format` kwarg.

    If running in Python >= 3.8, will attempt to add `force=True` to the kwargs
    for logging.basicConfig.

    After calling this method, use the kwarg `category` to pass in a value of
    `DataCategory` to denote data category. The default is `PRIVATE`. That is,
    if no changes are made to an existing set of log statements, the log output
    should be the same.

    The standard implementation of the logging API is a good reference:
    https://github.com/python/cpython/blob/3.9/Lib/logging/__init__.py
    """
    set_prefix(prefix)

    if "format" not in kwargs:
        kwargs["format"] = f"%(prefix)s{logging.BASIC_FORMAT}"

    # Ensure that all loggers created via `logging.getLogger` are instances of
    # the `ConfidentialLogger` class.
    logging.setLoggerClass(ConfidentialLogger)

    if len(logging.root.handlers) > 0:
        p = get_prefix()
        for line in _logging_basic_config_set_warning.splitlines():
            print(f"{p}{line}", file=sys.stderr)

    if "force" not in kwargs and sys.version_info >= (3, 8):
        kwargs["force"] = True

    old_root = logging.root

    root = ConfidentialLogger(logging.root.name)
    root.handlers = old_root.handlers

    logging.root = root
    logging.Logger.root = root  # type: ignore
    logging.Logger.manager = logging.Manager(root)  # type: ignore

    # https://github.com/kivy/kivy/issues/6733
    logging.basicConfig(**kwargs)


try:
    from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
    from pytorch_lightning.utilities import rank_zero_only
    PYTORCH_LIGHTNING_INSTALLED = True
except ImportError:
    LightningLoggerBase = None
    rank_zero_experiment = None
    rank_zero_only = None
    PYTORCH_LIGHTNING_INSTALLED = False

class ConfidentialConsoleLogger(LightningLoggerBase):  # type: ignore
    """
    Quick and dirty implementation of a PyTorch Lightning logger that plays
    nicely with compliance requirements (`SystemLog:` prefix), that can be
    used when inside detonation chambers.
    """

    def __init__(self):
        if not PYTORCH_LIGHTNING_INSTALLED:
            raise Exception("")

        super().__init__()
        self.logger = logging.getLogger(__name__)

    @property
    @rank_zero_experiment
    def experiment(self):
        return "confidential_experiment"

    @property
    def name(self):
        return "confidential_name"

    @property
    def run_id(self):
        return "confidential_run_id"

    @property
    def version(self):
        return "confidential_version"

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]):
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        for key, value in params.items():
            self.logger.info(f"Hyperparameter: {key} -> {value}", category=DataCategory.PUBLIC)
        else:
            self.logger.info("No hyperparameters to log.", category=DataCategory.PUBLIC)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for name, value in metrics.items():
            self.logger.info(
                f"Metric (step {step}): {name} -> {value}", category=DataCategory.PUBLIC
            )


# https://github.com/MicrosoftDocs/azure-docs/issues/66065




class ConfidentialParallelRunStepLogger(logging.Logger):
    """
    TODO:
    https://github.com/MicrosoftDocs/azure-docs/issues/66065
    https://docs.microsoft.com/en-us/azure/machine-learning/how-to-debug-parallel-run-step#how-do-i-log-from-my-user-script-from-a-remote-context

    How to use?

    ```
    def init():
        logger = ConfidentialParallelRunStepLogger(__name__)
    
    def run(mini_batch):
        # TODO: singleton pattern.
    ```
    """

    def __init__(self):
        super().__init__(__name__)
        try:
            from azureml_user.parallel_run import EntryScript  # type: ignore
            self.logger = EntryScript().logger
        except ImportError:
            # TODO: log some kind of warning.
            self.logger = logging.getLogger(__name__)

    def _log(self, level, msg, category, args, **kwargs):
        # TODO: EntryScript.logger...
        p = ""
        if category == DataCategory.PUBLIC:
            p = get_prefix()
        
        self.logger._log(level, msg, args, extra={"prefix": p}, **kwargs)
