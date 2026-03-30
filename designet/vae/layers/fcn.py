import torch.nn as nn


class FCN(nn.Module):
    def __init__(
        self,
        d_model,
        n_commands,
        args_dim=256,
    ):
        super().__init__()

        self.n_args = 8
        self.args_dim = args_dim

        self.command_fcn = nn.Linear(d_model, n_commands)

        self.args_fcn = nn.Linear(d_model, self.n_args)
        self.continuity_fcn = nn.Linear(d_model, 3)
        self.alignment_fcn = nn.Linear(d_model, 3)

    def forward(self, out):
        S, N, _ = out.shape
        res = {}

        command_logits = self.command_fcn(out)
        res["commands"] = command_logits

        args_logits = self.args_fcn(out)
        args_logits = args_logits.reshape(S, N, self.n_args)
        res["args"] = args_logits

        continuity_logits = self.continuity_fcn(out)
        res["continuity"] = continuity_logits

        alignment_logits = self.alignment_fcn(out)
        res["alignment"] = alignment_logits

        return res


class HierarchFCN(nn.Module):
    def __init__(self, d_model, dim_z):
        super().__init__()

        self.visibility_fcn = nn.Linear(d_model, 2)
        self.z_fcn = nn.Linear(d_model, dim_z)

    def forward(self, out):
        visibility_logits = self.visibility_fcn(out)
        z = self.z_fcn(out)

        return visibility_logits.unsqueeze(0), z.unsqueeze(0)
