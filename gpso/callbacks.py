"""
User defined callbacks.
"""

import logging
from enum import Enum, auto, unique


@unique
class CallbackTypes(Enum):
    """
    Define callback types.
    """

    post_initialise = auto()
    pre_iteration = auto()
    post_iteration = auto()
    post_update = auto()
    pre_finalise = auto()


class GPSOCallback:
    # do not forget to define type of the callback
    callback_type = None

    def __init__(self):
        # all arguments for callback needs to be defined here

        # when subclassing, it is recommended to call super().__init__() for
        # sanity check
        assert (
            self.callback_type in CallbackTypes
        ), "Callback type must be one of `CallbackTypes`"

    def run(self, optimiser):
        # run only takes one argument - the GPSOptimiser itself

        # when subclassing, it is recommended to call super().run() for sanity
        # check
        logging.info(f"Running {self.__class__.__name__} callback...")
