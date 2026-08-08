from torch import nn


class GateConv(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)