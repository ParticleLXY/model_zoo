from torch import nn


class ShortConnect(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)