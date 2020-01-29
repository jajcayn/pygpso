"""
Useful utilities.
"""
import json

JSON_EXT = ".json"
PKL_EXT = ".pkl"


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
