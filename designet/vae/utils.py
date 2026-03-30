import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from designet.difflib.tensor import SVGTensor


def _get_key_padding_mask(commands, seq_dim=0):
    with torch.no_grad():
        key_padding_mask = (commands == SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")).cumsum(dim=seq_dim) > 0

        if seq_dim == 0:
            return key_padding_mask.transpose(0, 1)
        return key_padding_mask


def _get_padding_mask(commands, seq_dim=0, extended=False):
    with torch.no_grad():
        padding_mask = (commands == SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")).cumsum(dim=seq_dim) == 0
        padding_mask = padding_mask.float()

        if extended:
            # padding_mask doesn't include the final EOS, extend by 1 position to include it in the loss
            S = commands.size(seq_dim)
            torch.narrow(padding_mask, seq_dim, 3, S - 3).add_(torch.narrow(padding_mask, seq_dim, 0, S - 3)).clamp_(
                max=1
            )

        if seq_dim == 0:
            return padding_mask.unsqueeze(-1)
        return padding_mask


def _get_visibility_mask(commands, seq_dim=0):
    S = commands.size(seq_dim)
    with torch.no_grad():
        visibility_mask = (commands == SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")).sum(dim=seq_dim) < S - 1

        if seq_dim == 0:
            return visibility_mask.unsqueeze(-1)
        return visibility_mask


def _get_key_visibility_mask(commands, seq_dim=0):
    S = commands.size(seq_dim)
    with torch.no_grad():
        key_visibility_mask = (commands == SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")).sum(dim=seq_dim) >= S - 1

        if seq_dim == 0:
            return key_visibility_mask.transpose(0, 1)
        return key_visibility_mask


def _sample_categorical(temperature=0.0001, *args_logits):
    if len(args_logits) == 1:
        (arg_logits,) = args_logits
        return Categorical(logits=arg_logits / temperature).sample()
    return (*(Categorical(logits=arg_logits / temperature).sample() for arg_logits in args_logits),)


def _threshold_sample(arg_logits, threshold=0.5, temperature=1.0):
    scores = F.softmax(arg_logits / temperature, dim=-1)[..., 1]
    return scores > threshold
