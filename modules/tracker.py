import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from accelerate.tracking import GeneralTracker, on_main_process
import os
from typing import Union
import numpy as np
import matplotlib.pyplot as plt


# 0. 自定义追踪器
class CustomTracker(GeneralTracker):
    """
    my custom `Tracker` class that supports `tensorboard`. Should be initialized at the start of your script.

    Args:
        run_name (`str`):
            The name of the experiment run
        logging_dir (`str`, `os.PathLike`):
            Location for TensorBoard logs to be stored.
        kwargs:
            Additional key word arguments passed along to the `tensorboard.SummaryWriter.__init__` method.
    """

    name = "tensorboard"
    requires_logging_directory = True

    @on_main_process
    def __init__(self, logging_dir: Union[str, os.PathLike],run_name: str, 
                 **kwargs):
        super().__init__()
        self.run_name = run_name
        self.logging_dir = os.path.join(logging_dir, run_name)
        self.writer = SummaryWriter(self.logging_dir, **kwargs)

    @property
    def tracker(self):
        return self.writer

    @on_main_process
    def add_scalar(self, tag, scalar_value, **kwargs):
        self.writer.add_scalar(tag=tag, scalar_value=scalar_value, **kwargs)

    @on_main_process
    def add_text(self, tag, text_string, **kwargs):
        self.writer.add_text(tag=tag, text_string=text_string, **kwargs)

    @on_main_process
    def add_figure(self, tag, figure, **kwargs):
        self.writer.add_figure(tag=tag, figure=figure, **kwargs)


