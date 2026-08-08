from torch import nn


class BaseNet(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        separator: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()

        self.encoder = encoder
        self.separator = separator
        self.decoder = decoder

    def forward(self, x):
        encoder_out = self.encoder(x)
        separator_out = self.separator(encoder_out)
        decoder_out = self.decoder(separator_out)

        return decoder_out
