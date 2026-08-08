from torch import nn


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        conv_type,
    ):
        super().__init__()