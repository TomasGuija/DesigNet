import copy

from torch.nn import MultiheadAttention
from torch.nn import functional as F
from torch.nn.modules.container import ModuleList
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.module import Module
from torch.nn.modules.normalization import LayerNorm


class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, memory2=None, mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, memory2=memory2, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """

    __constants__ = ["norm"]

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt,
        memory,
        memory2=None,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for mod in self.layers:
            output = mod(
                output,
                memory,
                memory2=memory2,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class TransformerEncoderLayerImproved(Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", d_global2=None, batch_first=True
    ):
        super(TransformerEncoderLayerImproved, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        if d_global2 is not None:
            self.linear_global2 = Linear(d_global2, d_model)

        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2_2 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerEncoderLayerImproved, self).__setstate__(state)

    def forward(self, src, memory2=None, src_mask=None, src_key_padding_mask=None):
        src1 = self.norm1(src)
        src2 = self.self_attn(src1, src1, src1, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)

        if memory2 is not None:
            src2_2 = self.linear_global2(memory2)
            src = src + self.dropout2_2(src2_2)

        src1 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src1))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayerGlobalImproved(Module):
    def __init__(
        self,
        d_model,
        d_global,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        d_global2=None,
        batch_first=True,
    ):

        super(TransformerDecoderLayerGlobalImproved, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        self.linear_global = Linear(d_global, d_model)

        if d_global2 is not None:
            self.linear_global2 = Linear(d_global2, d_model)

        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout2_2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerDecoderLayerGlobalImproved, self).__setstate__(state)

    def forward(self, tgt, memory, memory2=None, tgt_mask=None, tgt_key_padding_mask=None, *args, **kwargs):
        tgt1 = self.norm1(tgt)
        tgt2 = self.self_attn(tgt1, tgt1, tgt1, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.linear_global(memory)
        tgt = tgt + self.dropout2(tgt2)  # implicit broadcast

        if memory2 is not None:
            tgt2_2 = self.linear_global2(memory2)
            tgt = tgt + self.dropout2_2(tgt2_2)

        tgt1 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt1))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt
