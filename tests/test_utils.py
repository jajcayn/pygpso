"""
Set of tests for utilities within the GPSO.
"""

import json
import logging
import os
import re
import unittest
from shutil import rmtree

from gpso.utils import JSON_EXT, LOG_EXT, load_json, make_dirs, set_logger


class TestUtils(unittest.TestCase):
    TEMP_FOLDER = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "temp"
    )
    TEST_DICT = {"list1": ["a", "b", "c"], "float": 12.3, "string": "hello"}
    JSON_FILE = f"test{JSON_EXT}"
    LOG_FILE = f"log"
    LOGS_TO_WRITE = {
        "INFO": "test info",
        "DEBUG": "test debug",
        "WARNING": "test warning",
        "ERROR": "test error",
    }

    @classmethod
    def setUpClass(cls):
        os.makedirs(cls.TEMP_FOLDER)

    @classmethod
    def tearDownClass(cls):
        rmtree(cls.TEMP_FOLDER)

    def test_load_json(self):
        # create and save test json
        filename = os.path.join(self.TEMP_FOLDER, self.JSON_FILE)
        with open(filename, "w") as f:
            json.dump(self.TEST_DICT, f)

        self.assertDictEqual(load_json(filename), self.TEST_DICT)

    def test_make_dirs(self):
        # make new dir
        make_dirs(os.path.join(self.TEMP_FOLDER, "new_test_dir"))
        self.assertTrue(
            os.path.exists(os.path.join(self.TEMP_FOLDER, "new_test_dir"))
        )
        root_logger = logging.getLogger()
        with self.assertLogs(root_logger, level="WARNING") as cm:
            make_dirs(os.path.join(self.TEMP_FOLDER, "new_test_dir"))
        print(cm.output)
        self.assertTrue(
            "tests/temp/new_test_dir could not be created: [Errno 17] File"
            " exists:" in cm.output[0],
        )

    def test_logger(self):
        logname = os.path.join(self.TEMP_FOLDER, self.LOG_FILE)
        set_logger(log_filename=logname, log_level=logging.DEBUG)
        # write log file
        for line in self.LOGS_TO_WRITE:
            func = getattr(logging, line.lower())
            func(self.LOGS_TO_WRITE[line])
        # read log file
        log_read = open(logname + LOG_EXT).readlines()
        # check line by line
        for log_line, log_wanted in zip(log_read, self.LOGS_TO_WRITE):
            # find key in log_line
            log_ = re.findall(f"{log_wanted}: (.*)\\n", log_line)[0]
            self.assertEqual(log_, self.LOGS_TO_WRITE[log_wanted])


if __name__ == "__main__":
    unittest.main()
