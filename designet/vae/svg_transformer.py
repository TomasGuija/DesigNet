import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm

from designet.difflib.tensor import SVGTensor

from designet.vae.layers.fcn import FCN, HierarchFCN
from designet.vae.layers.positional_encoding import PositionalEncodingSinCos
from designet.vae.layers.transformer import (
    TransformerDecoder,
    TransformerDecoderLayerGlobalImproved,
    TransformerEncoder,
    TransformerEncoderLayerImproved,
)
from designet.vae.utils import (
    _get_key_padding_mask,
    _get_key_visibility_mask,
    _get_padding_mask,
    _get_visibility_mask,
    _sample_categorical,
    _threshold_sample,
)


class SVGTransformer(nn.Module):
    """
    Top-level Transformer-based VAE model for SVG generation.

    It combines a two-level encoder, latent sampling blocks, and a two-stage
    decoder that predicts path visibility plus per-token commands/arguments.
    The encoder produces both path-level and glyph-level latents, and the
    decoder is conditioned on both. Additionally, the model can predict continuity
    and line alignment logits for each token, which can be later used by the
    self-refinement modules.
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.encoder = TwoLevelEncoder(cfg)

        self.vae1 = VAEL(cfg)
        self.vae2 = VAEL(cfg)
        self.decoder = DecoderL(cfg)

        self.register_buffer("cmd_args_mask", SVGTensor.CMD_ARGS_MASK_8_ARGS)

    def forward(
        self,
        commands,
        args,
        z2=None,
        z1=None,
        visibility_logits=None,
        return_tgt=True,
        encode_mode=False,
    ):
        """
        Run the full model or only the encoder depending on flags.

        When ``encode_mode`` is True, this returns latent representations.
        Otherwise it returns decoder logits and optionally target tensors.
        """

        # Initialize latent stats for branches where `z` is provided externally.
        mu1 = logsigma1 = None
        mu2 = logsigma2 = None

        # If no latent representation is provided
        if z2 is None or z1 is None:
            # Encode the input
            z2, z1 = self.encoder(commands, args)
            z2, mu2, logsigma2 = self.vae2(z2)
            z1, mu1, logsigma1 = self.vae1(z1)

        if encode_mode:
            return z2, z1

        # output logits for commands, args and visibility
        z2 = z2.unsqueeze(-2)
        z1 = z1.unsqueeze(-2)
        out_logits = self.decoder(z2, z_path=z1, visibility_logits=visibility_logits)

        res = {
            "command_logits": out_logits["commands"],
            "args_logits": out_logits["args"],
            "visibility_logits": out_logits["visibility"],
        }

        res["cont_logits"] = out_logits["continuity"]
        res["alignment_logits"] = out_logits["alignment"]

        if return_tgt:
            res["tgt_commands"] = commands
            res["tgt_args"] = args

        res["mu2"] = mu2
        res["logsigma2"] = logsigma2
        res["mu1"] = mu1
        res["logsigma1"] = logsigma1

        return res

    @torch.no_grad()
    def greedy_sample(
        self,
        commands=None,
        args=None,
        z2=None,
        z1=None,
        hierarch_logits=None,
        close_paths=True,
        temperature=0.0001,
        pred=None,
    ):
        """
        Decode logits into command/argument predictions with greedy sampling.

        This method can decode from inputs (encode+decode) or directly from
        provided latent codes.
        """

        if pred is None:
            pred = self.forward(
                commands,
                args,
                z2=z2,
                z1=z1,
                visibility_logits=hierarch_logits,
                return_tgt=False,
            )

        commands_y, _ = _sample_categorical(temperature, pred["command_logits"], pred["args_logits"])
        args_y = pred["args_logits"]

        visibility_y = _threshold_sample(pred["visibility_logits"], threshold=0.7).bool().squeeze(-1)

        commands_y, args_y = self._make_valid(commands_y, args_y, visibility_y)

        if close_paths:
            args_y = self._apply_closing_midpoints(commands_y, args_y)

        return commands_y, args_y

    def _make_valid(self, commands_y, args_y, visibility_y=None, PAD_VAL=-1):
        """
        Post-process sampled tokens into a valid SVG-like sequence format.

        The routine enforces EOS semantics, applies visibility masking, and
        zeroes arguments that are invalid for each command type.
        """
        B, G, S, _ = args_y.shape

        if visibility_y is not None:
            S = commands_y.size(-1)
            commands_y[~visibility_y] = commands_y.new_tensor(
                [SVGTensor.COMMANDS_SIMPLIFIED.index("m"), *[SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")] * (S - 1)]
            )
            args_y[~visibility_y] = PAD_VAL

        # If there is more than one MOVE, convert it to EOS
        move_mask = commands_y == SVGTensor.COMMANDS_SIMPLIFIED.index("m")
        S = commands_y.size(-1)
        idxs = torch.arange(S, device=commands_y.device).view(1, 1, S).expand_as(commands_y)
        big = torch.full_like(idxs, S)
        first_move_idx = torch.where(move_mask, idxs, big).min(dim=-1).values  # (B,G)
        extra_moves = move_mask & (idxs > first_move_idx.unsqueeze(-1))
        commands_y = torch.where(
            extra_moves, commands_y.new_full((), SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")), commands_y
        )

        close_mask = commands_y == SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")
        cumsum_close = torch.cumsum(close_mask, dim=-1)
        after_first_close = cumsum_close >= 1
        commands_y = commands_y.masked_fill(after_first_close, SVGTensor.COMMANDS_SIMPLIFIED.index("EOS"))

        start_pos = args_y[..., :2]  # (B, G, S, 2)
        end_pos = args_y[..., -2:]  # (B, G, S, 2)

        # Default next-start
        next_start = torch.zeros_like(start_pos)
        next_start[..., :-1, :] = start_pos[..., 1:, :]

        averaged_endpoints = 0.5 * (end_pos + next_start)

        # Identify MOVE and EOS command positions
        move_mask = commands_y == SVGTensor.COMMANDS_SIMPLIFIED.index("m")  # (B, G, S)
        eos_mask = commands_y == SVGTensor.COMMANDS_SIMPLIFIED.index("EOS")  # (B, G, S)

        # Identify last valid command in each group
        valid_cmd_mask = ~eos_mask
        non_move_mask = ~move_mask
        valid_nonmove = valid_cmd_mask & non_move_mask

        idxs = torch.arange(S, device=commands_y.device).view(1, 1, -1).expand(B, G, S)
        cmd_idxs = idxs.masked_fill(~valid_nonmove, -1)
        last_valid_idx = cmd_idxs.max(dim=-1).values  # shape (B, G)

        valid_mask = idxs < last_valid_idx.unsqueeze(-1)  # (B,G,S)

        # Now update end only at allowed positions
        args_y[valid_mask][..., -2:] = averaged_endpoints[valid_mask]

        mask = self.cmd_args_mask[commands_y.long()].bool()
        args_y[~mask] = PAD_VAL

        # Skip start position
        args_y = args_y[..., -6:]

        return commands_y, args_y

    def _apply_closing_midpoints(self, commands_y, args_y):
        """Adjust start/end points so open contours close with smooth midpoints."""
        B, G, S, _ = args_y.shape
        move_endpoints = args_y[:, :, 0, -2:]
        endpoints = args_y[:, :, :, -2:]

        next_cmd = torch.zeros_like(commands_y)
        next_cmd[..., :-1] = commands_y[..., 1:]
        next_cmd[..., -1] = 3

        is_last_endpoint = (commands_y != 3) & (next_cmd == 3)
        idxs = torch.arange(S, device=commands_y.device)
        idxs_expanded = idxs.view(1, 1, S).expand(commands_y.shape)
        masked_idxs = idxs_expanded.masked_fill(~is_last_endpoint, -1)
        last_endpoint_idx = masked_idxs.max(dim=-1).values

        if (last_endpoint_idx >= 0).any():
            valid_mask = last_endpoint_idx >= 0
            batch_indices = torch.arange(B, device=args_y.device).unsqueeze(1).expand(B, G)
            group_indices = torch.arange(G, device=args_y.device).unsqueeze(0).expand(B, G)

            last_endpoint_idx_exp = last_endpoint_idx.clamp(min=0).unsqueeze(-1).unsqueeze(-1)
            last_endpoint_idx_exp = last_endpoint_idx_exp.expand(-1, -1, -1, 2)
            last_endpoints = endpoints.gather(dim=2, index=last_endpoint_idx_exp).squeeze(2)

            midpoint = (move_endpoints + last_endpoints) / 2
            midpoint = midpoint[valid_mask]

            args_y[batch_indices[valid_mask], group_indices[valid_mask], 0, -2:] = midpoint
            args_y[batch_indices[valid_mask], group_indices[valid_mask], last_endpoint_idx[valid_mask], -2:] = midpoint

        return args_y


class TwoLevelEncoder(nn.Module):
    """Hierarchical encoder that builds path-level and glyph-level latents."""

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        seq_len = cfg["max_seq_len"]
        self.embedding = SVGEmbeddingL(cfg, seq_len)

        encoder_layer = TransformerEncoderLayerImproved(
            cfg["d_model"], cfg["n_heads"], cfg["dim_feedforward"], cfg["dropout"]
        )
        encoder_norm = LayerNorm(cfg["d_model"])
        self.encoder = TransformerEncoder(encoder_layer, cfg["n_layers"], encoder_norm)

        self.hierarchical_PE = PositionalEncodingSinCos(cfg["d_model"], max_len=cfg["max_num_groups"])

        hierarchical_encoder_layer = TransformerEncoderLayerImproved(
            cfg["d_model"], cfg["n_heads"], cfg["dim_feedforward"], cfg["dropout"]
        )
        hierarchical_encoder_norm = LayerNorm(cfg["d_model"])
        self.hierarchical_encoder = TransformerEncoder(
            hierarchical_encoder_layer, cfg["n_layers"], hierarchical_encoder_norm
        )

    def forward(self, commands, args):
        """Encode batched command/argument tensors into latent embeddings."""
        N, G, _ = commands.shape

        # Masking groups
        visibility_mask = _get_visibility_mask(commands, seq_dim=-1)
        key_visibility_mask = _get_key_visibility_mask(commands, seq_dim=-1)

        # Flatten [G, S, N, ...] → [N * G, S, ...] so each group in the batch is treated as an independent sequence
        # This allows parallel processing of all groups across all samples in a single transformer/layer pass
        # We compute a mask for detecting padding values in each sequence
        commands, args = commands.flatten(0, 1), args.flatten(0, 1)
        padding_mask = _get_padding_mask(commands, seq_dim=-1)
        key_padding_mask = _get_key_padding_mask(commands, seq_dim=-1)

        src1 = self.embedding(commands, args)

        # This is the first stage of the encoding. It will provide an embedding for each path of each sample
        memory = self.encoder(src1, mask=None, src_key_padding_mask=key_padding_mask)

        # "e perform Average Pooling over the embeddings of each sequence of a group/path
        divisor = padding_mask[..., 1:-1].sum(dim=-1, keepdim=True).clamp(min=1)
        z1 = (memory[..., 1:-1, :] * padding_mask[..., 1:-1, None]).sum(dim=-2) / divisor

        z1 = z1.unflatten(0, (N, G))

        # Second stage of Encoding
        src2 = z1

        # Apply an ordered positional encoding to inform the model about the original order of paths for each sample
        src2 = self.hierarchical_PE(src2)

        # Second encoder
        memory = self.hierarchical_encoder(src2, mask=None, src_key_padding_mask=key_visibility_mask)
        z2 = (memory * visibility_mask.unsqueeze(-1)).sum(dim=-2) / visibility_mask.sum(dim=-1, keepdim=True)

        return z2, z1


class VAEL(nn.Module):
    """Gaussian latent head with reparameterization during training."""

    def __init__(self, cfg):
        super(VAEL, self).__init__()

        self.enc_mu_fcn = nn.Linear(cfg["d_model"], cfg["dim_z"])
        self.enc_sigma_fcn = nn.Linear(cfg["d_model"], cfg["dim_z"])

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.normal_(self.enc_mu_fcn.weight, std=0.001)
        nn.init.constant_(self.enc_mu_fcn.bias, 0)
        nn.init.normal_(self.enc_sigma_fcn.weight, std=0.001)
        nn.init.constant_(self.enc_sigma_fcn.bias, 0)

    def forward(self, z):
        """Predict mean/log-variance and sample latent vectors."""
        mu, logsigma = self.enc_mu_fcn(z), self.enc_sigma_fcn(z)
        sigma = torch.exp(logsigma / 2.0)
        if self.training:
            # Apply the re-parametrization trick only during training
            z = mu + sigma * torch.randn_like(sigma)
        else:
            # During inference, we simply use the mean of the latent distribution as the output
            z = mu
        return z, mu, logsigma


class DecoderL(nn.Module):
    """Two-stage decoder for group visibility and token prediction."""

    def __init__(self, cfg):
        super(DecoderL, self).__init__()

        self.cfg = cfg

        hierarchical_decoder_layer = TransformerDecoderLayerGlobalImproved(
            cfg["d_model"],
            cfg["dim_z"],
            cfg["n_heads"],
            cfg["dim_feedforward"],
            cfg["dropout"],
        )
        hierarchical_decoder_norm = LayerNorm(cfg["d_model"])
        self.hierarchical_decoder = TransformerDecoder(
            hierarchical_decoder_layer, cfg["n_layers_decode"], hierarchical_decoder_norm
        )

        self.hierarchical_fcn = HierarchFCN(cfg["d_model"], cfg["dim_z"])

        seq_len = cfg["max_seq_len"] + 1
        self.embedding = ConstEmbeddingL(cfg, seq_len)

        decoder_layer = TransformerDecoderLayerGlobalImproved(
            cfg["d_model"], cfg["dim_z"], cfg["n_heads"], cfg["dim_feedforward"], cfg["dropout"]
        )
        decoder_norm = LayerNorm(cfg["d_model"])
        self.decoder = TransformerDecoder(decoder_layer, cfg["n_layers_decode"], decoder_norm)

        self.fcn = FCN(
            cfg["d_model"],
            cfg["n_commands"],
        )

    def forward(self, z, z_path=None, visibility_logits=None):
        """Decode latent codes into visibility, command, and argument logits."""
        N = z.shape[0]
        G = self.cfg["num_groups_proposal"]

        # First decoding stage
        # Hierarchical decoding (D^2)
        # Decoder layers are conditioned on global latent z and optionally label embeddings
        # Outputs for each group simultaneously, not sequentially
        if visibility_logits is None:
            z_path = z_path.squeeze(-2)
            out = self.hierarchical_decoder(z_path, z, tgt_mask=None, tgt_key_padding_mask=None)
            # From the output of the first decoder, we apply two FFNN. One predicts the visibility logits for each group
            # The other a new embedding for each of these groups
            visibility_logits, z = self.hierarchical_fcn(out)

        visibility_logits, z = visibility_logits.squeeze(0), z.squeeze(0)

        # Second stage of the decoder
        # z contains one embedding per group
        # src -> positional encoding
        z_flattened = z.flatten(0, 1)
        src = self.embedding(z_flattened)
        # out -> output of second decoder
        out = self.decoder(src, z_flattened.unsqueeze(-2), tgt_mask=None, tgt_key_padding_mask=None)
        out_logits = self.fcn(out)
        out_logits = {k: v.unflatten(0, (N, G)) for k, v in out_logits.items()}
        out_logits["visibility"] = visibility_logits

        return out_logits


class ConstEmbeddingL(nn.Module):
    """Positional encoding used as decoder input template."""

    def __init__(self, cfg, seq_len):
        super().__init__()
        self.cfg = cfg
        self.seq_len = seq_len
        self.PE = PositionalEncodingSinCos(cfg["d_model"], max_len=seq_len)

    def forward(self, z):
        N = z.size(0)
        src = self.PE(z.new_zeros(N, self.seq_len, self.cfg["d_model"]))
        return src


class SVGEmbeddingL(nn.Module):
    """Token embedding block for SVG commands and command arguments."""

    def __init__(self, cfg, seq_len):
        super().__init__()

        self.cfg = cfg
        self.pad_val = -1

        self.command_embed = nn.Embedding(cfg["n_commands"], cfg["d_model"], padding_idx=4)

        # ---- Arguments branch ----
        self.arg_conv = nn.Conv1d(in_channels=8, out_channels=cfg["d_model"], kernel_size=1)

        self.pos_encoding = PositionalEncodingSinCos(cfg["d_model"], max_len=seq_len + 2)
        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.command_embed.weight, mode="fan_in")
        nn.init.kaiming_normal_(self.arg_conv.weight, mode="fan_in")

    def forward(self, commands, args):
        """Embed command ids and arguments into transformer token features."""
        GN, S = commands.shape
        cmd_embeddings = self.command_embed(commands.long())

        mask = (args != self.pad_val).float()
        args_masked = args * mask
        args_embeddings = self.arg_conv(args_masked.flatten(0, 1).unsqueeze(-1))
        args_embeddings = args_embeddings.reshape(GN, S, -1)

        src = cmd_embeddings + args_embeddings
        src = self.pos_encoding(src)
        return src
