from __future__ import annotations

import torch
from torch import Tensor


class SVGTensor:
    #                       0    1    2    3     4
    COMMANDS_SIMPLIFIED = ["m", "l", "c", "EOS", "SOS"]

    CMD_ARGS_MASK = torch.tensor(
        [
            [0, 0, 0, 0, 1, 1],  # m
            [0, 0, 0, 0, 1, 1],  # l
            [1, 1, 1, 1, 1, 1],  # c
            [0, 0, 0, 0, 0, 0],  # EOS
            [0, 0, 0, 0, 0, 0],  # SOS
        ]
    )

    CMD_ARGS_MASK_8_ARGS = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 1, 1],  # m
            [1, 1, 0, 0, 0, 0, 1, 1],  # l
            [1, 1, 1, 1, 1, 1, 1, 1],  # c
            [0, 0, 0, 0, 0, 0, 0, 0],  # EOS
            [0, 0, 0, 0, 0, 0, 0, 0],  # SOS
        ]
    )

    class Index:
        COMMAND = 0
        START_POS = slice(1, 3)
        CONTROL1 = slice(3, 5)
        CONTROL2 = slice(5, 7)
        END_POS = slice(7, 9)

    class IndexArgs:
        CONTROL1 = slice(0, 2)
        CONTROL2 = slice(2, 4)
        END_POS = slice(4, 6)

    position_keys = ["control1", "control2", "end_pos"]
    all_position_keys = ["start_pos", *position_keys]
    arg_keys = [*position_keys]
    all_arg_keys = ["start_pos", *arg_keys]
    cmd_arg_keys = ["commands", *arg_keys]
    all_keys = ["commands", *all_arg_keys]

    def __init__(
        self,
        commands: Tensor,
        control1: Tensor,
        control2: Tensor,
        end_pos: Tensor,
        seq_len=None,
        PAD_VAL: int = -1,
        ARGS_DIM: int = 256,
        filling: int = 0,
    ):

        self.commands = commands.reshape(-1, 1).float()

        self.control1 = control1.float()
        self.control2 = control2.float()
        self.end_pos = end_pos.float()

        dev = commands.device
        self.seq_len = torch.tensor(len(commands), device=dev) if seq_len is None else seq_len

        self.PAD_VAL = PAD_VAL
        self.ARGS_DIM = ARGS_DIM

        self.sos_token = torch.Tensor([self.COMMANDS_SIMPLIFIED.index("SOS")], device=dev).unsqueeze(-1)
        self.pad_token = torch.Tensor([self.COMMANDS_SIMPLIFIED.index("EOS")], device=dev).unsqueeze(-1)
        self.eos_token = self.pad_token
        self.filling = filling

    @property
    def start_pos(self):
        start_pos = self.end_pos[:-1]

        return torch.cat([start_pos.new_zeros(1, 2), start_pos])

    @staticmethod
    def from_data(data, *args, **kwargs):
        return SVGTensor(
            data[:, SVGTensor.Index.COMMAND],
            data[:, SVGTensor.Index.CONTROL1],
            data[:, SVGTensor.Index.CONTROL2],
            data[:, SVGTensor.Index.END_POS],
            *args,
            **kwargs,
        )

    @staticmethod
    def from_cmd_args(commands, args, *nargs, **kwargs):
        try:
            svg = SVGTensor(
                commands,
                args[:, SVGTensor.IndexArgs.CONTROL1],
                args[:, SVGTensor.IndexArgs.CONTROL2],
                args[:, SVGTensor.IndexArgs.END_POS],
                *nargs,
                **kwargs,
            )

            # We try to access data to make sure the SVGTensor was created successfully.
            _ = svg.data

            return svg
        except Exception as _:  # noqa
            return None

    def get_data(self, keys):
        return torch.cat([self.__getattribute__(key) for key in keys], dim=-1)

    @property
    def data(self):
        return self.get_data(self.all_keys)

    def add_sos(self):
        self.commands = torch.cat([self.sos_token, self.commands])

        for key in self.arg_keys:
            v = self.__getattribute__(key)
            self.__setattr__(key, torch.cat([v.new_full((1, v.size(-1)), self.PAD_VAL), v]))

        self.seq_len += 1
        return self

    def drop_sos(self):
        for key in self.cmd_arg_keys:
            self.__setattr__(key, self.__getattribute__(key)[1:])

        self.seq_len -= 1
        return self

    def add_eos(self):
        self.commands = torch.cat([self.commands, self.eos_token])

        for key in self.arg_keys:
            v = self.__getattribute__(key)
            self.__setattr__(key, torch.cat([v, v.new_full((1, v.size(-1)), self.PAD_VAL)]))

        return self

    def pad(self, seq_len=51):
        pad_len = max(seq_len - len(self.commands), 0)

        self.commands = torch.cat([self.commands, self.pad_token.repeat(pad_len, 1)])

        for key in self.arg_keys:
            v = self.__getattribute__(key)
            self.__setattr__(key, torch.cat([v, v.new_full((pad_len, v.size(-1)), self.PAD_VAL)]))

        return self

    def cmds(self):
        return self.commands.reshape(-1)

    def args(self, with_start_pos=False):
        if with_start_pos:
            return self.get_data(self.all_arg_keys)

        return self.get_data(self.arg_keys)
