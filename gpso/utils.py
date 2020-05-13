"""
Useful utilities.
"""

import json
import logging
import os
from enum import Enum, unique

LOG_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_EXT = ".log"
JSON_EXT = ".json"
PKL_EXT = ".pkl"
H5_EXT = ".h5"


@unique
class PointLabels(Enum):
    """
    Helper for leaf labels - empty, evaluated using LSBM or GPR-based.
    """

    not_assigned = 0  # when no score is yet known
    evaluated = 1  # objective function is evaluated at the centre of the leaf
    gp_based = 2  # score was obtained using UCB via GP


def make_dirs(path):
    """
    Create directory.

    :param path: path for new directory
    :type path: str
    """
    try:
        os.makedirs(path)
    except OSError as error:
        logging.warning(f"{path} could not be created: {error}")


def load_json(filename):
    """
    Load JSON file.

    :param filename: filename for JSON file
    :type filename: str
    :return: loaded JSON data as nested dictionary
    :rtype: dict
    """
    with open(filename, "r") as file_handler:
        json_data = json.load(file_handler)

    return json_data


def set_logger(log_filename=None, log_level=logging.INFO):
    """
    Prepare logger.

    :param log_filename: filename for the log, if None, will not use logger
    :type log_filename: str|None
    :param log_level: logging level
    :type log_level: int
    """
    formatting = "[%(asctime)s] %(levelname)s: %(message)s"
    log_formatter = logging.Formatter(formatting, LOG_DATETIME_FORMAT)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers = []

    # set terminal logging
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # set file logging
    if log_filename is not None:
        if not log_filename.endswith(LOG_EXT):
            log_filename += LOG_EXT
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)
