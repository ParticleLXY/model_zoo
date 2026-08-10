from torch import nn


class EncoderBlock(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            kernel_size: list,
            conv_type: nn.Module = nn.Conv2d,
            act: nn.Module = nn.PReLU,
            norm: nn.Module = nn.BatchNorm2d,
        ):
        super().__init__()
        self.conv = conv_type(input_dim, output_dim, kernel_size)
        self.act = act()
        self.norm = norm(output_dim)

    def forward(self, x):
        conv_out = self.conv(x)
        norm_out = self.norm(conv_out)
        y = self.act(norm_out)
        return y



class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: list,
        output_dim: list,
        kernel_size: list,
        conv_type: nn.Module,
        act,
        norm,
    ):
        super().__init__()
        if len(input_dim) != len(output_dim):
            raise ValueError(
                "input_dim and output_dim must contain the same number of channels"
            )

        self.encoder = nn.ModuleList([
            EncoderBlock(
                input_channels,
                output_channels,
                kernel_size,
                conv_type,
                act,
                norm,
            )
            for input_channels, output_channels in zip(input_dim, output_dim)
        ])

    def forward(self, x):
        outputs = []
        for block in self.encoder:
            x = block(x)
            outputs.append(x)
        return outputs


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim: list,
        output_dim: list,
        kernel_size: list,
        conv_type: nn.Module,
        act,
        norm,
        short_connect: nn.Module,
    ):
        super().__init__()
        if len(input_dim) != len(output_dim):
            raise ValueError(
                "input_dim and output_dim must contain the same number of channels"
            )

        self.decoder = nn.ModuleList([
            EncoderBlock(
                input_channels,
                output_channels,
                kernel_size,
                conv_type,
                act,
                norm,
            )
            for input_channels, output_channels in zip(input_dim, output_dim)
        ])
        self.short_connect = short_connect

    def forward(self, x, encoder_out):
        if len(encoder_out) != len(self.decoder):
            raise ValueError(
                "encoder_out and decoder must contain the same number of blocks, "
                f"but got {len(encoder_out)} and {len(self.decoder)}"
            )

        for block, skip_out in zip(self.decoder, reversed(encoder_out)):
            x = self.short_connect(skip_out, x)
            x = block(x)
        return x
