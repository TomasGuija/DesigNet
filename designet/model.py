import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from .checkpoint import resolve_checkpoint_path
from .vae.svg_transformer import SVGTransformer
from .vae.utils import _sample_categorical, _threshold_sample


class FontConditionalSVGTransformer(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="designet",
    repo_url="https://github.com/TomasGuija/DesigNet",
    tags=["svg", "font-generation"],
):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.base_model = SVGTransformer(cfg)

        base_model_ckpt = cfg.get("checkpoint_path", None)
        if base_model_ckpt:
            base_model_ckpt = resolve_checkpoint_path(base_model_ckpt)
            state = torch.load(base_model_ckpt, map_location="cpu", weights_only=False)
            state_dict = state["state_dict"]
            clean_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
            self.base_model.load_state_dict(clean_state_dict)

        self.num_encoding_letters = cfg["num_encoding_glyphs"]
        self.num_decoding_letters = cfg["num_decoding_glyphs"]
        self.num_letters = self.num_encoding_letters + self.num_decoding_letters

        self.condition_letter_proj = nn.Linear(2 * cfg["dim_z"], cfg["dim_z"])

        self.condition_letter_proj_group = nn.Linear(2 * cfg["dim_z"], cfg["dim_z"])
        self.cls_token_group = nn.Parameter(torch.zeros(1, 1, cfg["dim_z"]))
        nn.init.normal_(self.cls_token_group, mean=0.0, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg["dim_z"],
            nhead=cfg["n_heads"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
            batch_first=True,
        )
        self.aggregator_transformer_group = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg["dim_z"]))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg["dim_z"],
            nhead=cfg["n_heads"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
            batch_first=True,
        )

        self.aggregator_transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.glyph_emb = nn.Embedding(self.num_encoding_letters + self.num_decoding_letters, cfg["dim_z"])

    def forward(
        self,
        input_commands,
        input_args,
        return_tgt=True,
        sel_self=None,
        sel_cross=None,
    ):
        """
        input_commands, input_args: List of glyph tensors
        """
        B, N, G, S = input_commands.shape

        assert (
            N == self.num_encoding_letters + self.num_decoding_letters
        ), "Number of glyphs must be equal to number of encoding + decoding glyphs."

        device = input_commands.device

        encoding_commands = input_commands[:, : self.num_encoding_letters, :, :].flatten(0, 1)
        encoding_args = input_args[:, : self.num_encoding_letters, :, :, :].flatten(0, 1)

        if sel_self is None:
            sel_self = torch.arange(self.num_encoding_letters, device=device).unsqueeze(0).expand(B, -1)
        if sel_cross is None:
            sel_cross = torch.arange(self.num_decoding_letters, device=device).unsqueeze(0).expand(B, -1)

        abs_self = sel_self
        abs_cross = sel_cross + self.num_encoding_letters
        abs_all = torch.cat([abs_self, abs_cross], dim=-1).to(device)

        glyph_embedding = self.glyph_emb(abs_all)

        # We only encode encoding glyphs
        z, z1 = self.base_model.encoder(encoding_commands, encoding_args)
        z1 = z1.squeeze(0).reshape(B, self.num_encoding_letters, G, -1)

        cls_token_group = self.cls_token_group[None].expand(B, 1, G, -1)
        z1_with_cls = torch.cat([cls_token_group, z1], dim=-3)
        z1_with_cls = z1_with_cls.transpose(1, 2).flatten(0, 1)
        z1_agg = self.aggregator_transformer_group(z1_with_cls)
        z1_agg = z1_agg[:, 0]
        z1_agg = z1_agg.reshape(B, G, -1)
        z1, mu1, logsigma1 = self.base_model.vae1(z1_agg)

        z1 = z1.unsqueeze(-3)
        z1 = z1.expand(-1, abs_all.size(-1), -1, -1)

        glyph_embedding_group = glyph_embedding.unsqueeze(-2)
        glyph_embedding_group = glyph_embedding_group.expand_as(z1)

        fused = torch.cat([z1, glyph_embedding_group], dim=-1)

        conditioned_group_embeddings = self.condition_letter_proj_group(fused)
        conditioned_group_embeddings = conditioned_group_embeddings.flatten(0, 1)

        # We take our latent representations from each encoding letters, aggregate them for each font and sample from latent space
        z = z.view(B, self.num_encoding_letters, -1)
        cls_token = self.cls_token.expand(B, -1, -1)
        z_with_cls = torch.cat([cls_token, z], dim=1)
        z_agg = self.aggregator_transformer(z_with_cls)
        z = z_agg[:, 0]  # take the first token

        z, mu, logsigma = self.base_model.vae2(z)

        z_exp = z.unsqueeze(1).expand(-1, abs_all.size(1), -1)
        concat_embeddings = torch.cat([z_exp, glyph_embedding], dim=-1)
        conditioned_embeddings = self.condition_letter_proj(concat_embeddings)
        conditioned_embeddings = conditioned_embeddings.flatten(0, 1).unsqueeze(1)

        out_logits = self.base_model.decoder(
            conditioned_embeddings,
            z_path=conditioned_group_embeddings,
        )

        out_logits = {k: v.unflatten(0, (B, N)) for k, v in out_logits.items()}
        res = {}

        res["mu1"] = mu1
        res["logsigma1"] = logsigma1
        res["mu2"] = mu
        res["logsigma2"] = logsigma

        k1 = sel_self.size(1)
        k2 = sel_cross.size(1)

        def split_logits(x):
            x_self = x[:, :k1].reshape(B * k1, *x.shape[2:])
            x_cross = x[:, k1:].reshape(B * k2, *x.shape[2:])
            return x_self, x_cross

        res["self_command_logits"], res["cross_command_logits"] = split_logits(out_logits["commands"])
        res["self_args_logits"], res["cross_args_logits"] = split_logits(out_logits["args"])

        res["self_cont_logits"], res["cross_cont_logits"] = split_logits(out_logits["continuity"])
        res["self_alignment_logits"], res["cross_alignment_logits"] = split_logits(out_logits["alignment"])

        res["self_visibility_logits"], res["cross_visibility_logits"] = split_logits(out_logits["visibility"])

        if return_tgt:

            def batched_take(x, idx):
                b = torch.arange(x.size(0), device=x.device).unsqueeze(-1)
                return x[b, idx]

            cmds = input_commands
            args = input_args

            res["tgt_commands_self"] = batched_take(cmds[:, : self.num_encoding_letters], sel_self)[..., 1:].reshape(
                -1, G, S - 1
            )
            res["tgt_args_self"] = batched_take(args[:, : self.num_encoding_letters], sel_self)[..., 1:, :].reshape(
                -1, G, S - 1, args.size(-1)
            )
            res["tgt_commands_cross"] = batched_take(cmds[:, self.num_encoding_letters :], sel_cross)[..., 1:].reshape(
                -1, G, S - 1
            )
            res["tgt_args_cross"] = batched_take(args[:, self.num_encoding_letters :], sel_cross)[..., 1:, :].reshape(
                -1, G, S - 1, args.size(-1)
            )

        return res

    def greedy_sample(
        self,
        input_cmds=None,
        input_args=None,
        close_paths=True,
        return_continuity=False,
        sel_self=None,
        sel_cross=None,
        pred=None,
        temperature=0.0001,
    ):
        if pred is None:
            pred = self.forward(
                input_cmds,
                input_args,
                return_tgt=False,
                sel_self=sel_self,
                sel_cross=sel_cross,
            )

        commands_y_cross, _ = _sample_categorical(temperature, pred["cross_command_logits"], pred["cross_args_logits"])
        args_y_cross = pred["cross_args_logits"]

        visibility_y_cross = _threshold_sample(pred["cross_visibility_logits"], threshold=0.5).bool().squeeze(-1)

        commands_y_cross, args_y_cross = self.base_model._make_valid(commands_y_cross, args_y_cross, visibility_y_cross)

        if close_paths:
            args_y_cross = self.base_model._apply_closing_midpoints(commands_y_cross, args_y_cross)

        commands_y_self, _ = _sample_categorical(temperature, pred["self_command_logits"], pred["self_args_logits"])
        args_y_self = pred["self_args_logits"]

        visibility_y_self = _threshold_sample(pred["self_visibility_logits"], threshold=0.7).bool().squeeze(-1)

        commands_y_self, args_y_self = self.base_model._make_valid(commands_y_self, args_y_self, visibility_y_self)

        if close_paths:
            args_y_self = self.base_model._apply_closing_midpoints(commands_y_self, args_y_self)

        if return_continuity:
            continuity_y_self = torch.argmax(pred["self_cont_logits"], dim=-1)
            continuity_y_cross = torch.argmax(pred["cross_cont_logits"], dim=-1)

            return commands_y_cross, args_y_cross, continuity_y_cross, commands_y_self, args_y_self, continuity_y_self

        return commands_y_cross, args_y_cross, commands_y_self, args_y_self
