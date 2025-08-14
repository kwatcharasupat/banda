import torch
from torch import nn
from banda.utils.registry import MetricRegistry

import torchmetrics as tm

@MetricRegistry.register()
class SignalNoiseRatio(tm.SignalNoiseRatio):
    pass

