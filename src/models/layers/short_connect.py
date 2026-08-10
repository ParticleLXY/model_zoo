import torch
from torch import nn


class ShortConnect(nn.Module):
    """Connect an encoder feature to a decoder feature."""

    def __init__(self, connect_type: str | None = "add", dim: int = 1):
        super().__init__()
        connect_type = "none" if connect_type is None else connect_type.lower()
        if connect_type not in {"add", "concat", "none"}:
            raise ValueError(
                f"Unsupported connect_type {connect_type!r}; "
                "expected 'add', 'concat', or 'none'"
            )

        self.connect_type = connect_type
        self.dim = dim
        self.connect = getattr(self, f"_{self.connect_type}")

    def _add(self, encoder_out: torch.Tensor, decoder_out: torch.Tensor):
        if encoder_out.shape != decoder_out.shape:
            raise ValueError(
                "add requires encoder_out and decoder_out to have the same shape, "
                f"but got {tuple(encoder_out.shape)} and {tuple(decoder_out.shape)}"
            )
        return encoder_out + decoder_out

    def _concat(self, encoder_out: torch.Tensor, decoder_out: torch.Tensor):
        if encoder_out.ndim != decoder_out.ndim:
            raise ValueError(
                "encoder_out and decoder_out must have the same number of dimensions, "
                f"but got {encoder_out.ndim} and {decoder_out.ndim}"
            )
        if not -encoder_out.ndim <= self.dim < encoder_out.ndim:
            raise IndexError(
                f"dim {self.dim} is out of range for {encoder_out.ndim}-D tensors"
            )
        dim = self.dim % encoder_out.ndim
        encoder_shape = encoder_out.shape[:dim] + encoder_out.shape[dim + 1:]
        decoder_shape = decoder_out.shape[:dim] + decoder_out.shape[dim + 1:]
        if encoder_shape != decoder_shape:
            raise ValueError(
                "concat requires all dimensions except the concatenation dimension "
                f"to match, but got {tuple(encoder_out.shape)} and "
                f"{tuple(decoder_out.shape)} for dim={self.dim}"
            )
        return torch.cat((encoder_out, decoder_out), dim=dim)

    @staticmethod
    def _none(encoder_out: torch.Tensor | None, decoder_out: torch.Tensor):
        return decoder_out

    def forward(
        self,
        encoder_out: torch.Tensor | None,
        decoder_out: torch.Tensor,
    ):
        return self.connect(encoder_out, decoder_out)
