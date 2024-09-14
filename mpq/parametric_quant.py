import abc
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn


class STECeil(torch.autograd.Function):
    """Straight-Through Estimator for the ceil function.
    Described in the Sony MPQ paper, originally from Bengio et al. 2013."""

    @staticmethod
    def forward(ctx, x: Tensor):
        return torch.ceil(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore
        return grad_output


class STERound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore
        return grad_output


@dataclass
class QuantParamRange:
    init: float
    min: float
    max: float


class Quantizer(nn.Module):
    def __init__(
        self,
        qmax: QuantParamRange,
        step: QuantParamRange,
        signed: bool,
        b_min: int = 2,
        fix_parameters: bool = False,
    ) -> None:
        from torch.nn.parameter import Parameter

        super().__init__()
        grad = not fix_parameters
        self.qmax = Parameter(torch.tensor(qmax.init), requires_grad=grad)
        self.qmax_min, self.qmax_max = qmax.min, qmax.max
        self.step = Parameter(torch.tensor(step.init), requires_grad=grad)
        self.step_min, self.step_max = step.min, step.max
        self.signed = signed
        self.b_min = b_min
        self.activated = True

    def set_mode(self, mode: str):
        if mode in ["fixed", "bypass"]:
            self.qmax.requires_grad = False
            self.step.requires_grad = False
            if mode == "bypass":
                self.activated = False
        elif mode == "mpq":
            self.qmax.requires_grad = True
            self.step.requires_grad = True

    @torch.no_grad()
    def _clip_params(self):
        # Ensure that stepsize is in specified range and a power of two
        step = torch.clamp(self.step, self.step_min, self.step_max)
        assert step.item() > 0  # HACK: step can be 0 without this line? Why?
        step = 2 ** torch.round(torch.log2(step))
        self.step.data.copy_(step)
        assert self.step.item() > 0
        # Ensure that dynamic range is in specified range
        qmax = torch.clamp(self.qmax, self.qmax_min, self.qmax_max)
        self.qmax.data.copy_(qmax)

    def b(self) -> Tensor:
        if not self.activated:
            # In bypass mode no quantization happens, so we return 32 bit.
            # Returning a constant ensures that gradient on this value is 0.
            # TODO: we return 32 bits, but this is float32, not int32.
            # We don't have a way to distinguish float{n} and int{n}.
            return torch.tensor(32.0)
        # Calculate the bitwidth from the step size and max value.
        self._clip_params()
        # Compute real `qmax`. According to Sony impl, this step is only done when reading `b`.
        qmax = torch.round(self.qmax / self.step) * self.step
        b = cast(Tensor, STECeil.apply(torch.log2(qmax / self.step + 1.0)))
        # we do not clip to `cfg.w_bitwidth_max` as qmax/d_q could correspond to more than 8 bit
        # TODO: network_size_weights() in original impl does not consider self.signed
        # and just add 1 to the bitwidth. Should we do the same?
        return torch.clamp(b + int(self.signed), min=self.b_min)

    def forward(self, x: Tensor):
        # if mode is "bypass" we don't want to do quantization
        if not self.activated:
            return x
        self._clip_params()
        qmin = -self.qmax if self.signed else torch.zeros_like(self.step)
        return self.step * STERound.apply(torch.clamp(x, qmin, self.qmax) / self.step)

    def extra_repr(self) -> str:
        return f"d={self.step.data}, qmax={self.qmax.data}, b(derived)={self.b()}"


def get_weight_quantizers(weight: Tensor, bias: Tensor | None):
    def find_step(param: Tensor, bits: int):
        maxabs_w = param.abs().max() + np.finfo(np.float32).eps
        wmax_to_intmax = torch.log2(maxabs_w / (2 ** (bits - 1) - 1))
        wmax_to_intmax = wmax_to_intmax.ceil() if bits > 4 else wmax_to_intmax.floor()
        return (2**wmax_to_intmax).item()

    weight_bits = 4
    step_init = find_step(weight, weight_bits)
    qmax_init = step_init * (2 ** (weight_bits - 1) - 1)
    qmax = QuantParamRange(qmax_init, 2**-8, 127)
    step = QuantParamRange(step_init, 2**-8, 1)
    # TODO: should probably find step for bias as well
    # TODO: shouldn't bias quantization follow activation quantization?
    quant_w = Quantizer(qmax, step, signed=True)
    if bias is not None:
        quant_b = Quantizer(qmax, step, signed=True)
        return quant_w, quant_b
    else:
        return quant_w, None


class MPQModule(nn.Module):
    @abc.abstractmethod
    def get_weight_bytes(self) -> Tensor:
        pass

    @abc.abstractmethod
    def get_activ_bytes(self) -> Tensor:
        pass


class MPQConv2d(MPQModule):
    r"""Quantized Convolution.

    Quantized Convolution where the input/output
    relationship is

    .. math::

        y_{n, a, b} = \sum_{m} \sum_{i} \sum_{j} Q_w(w_{n, m, i, j}) x_{m, a + i, b + j} + Q_b(b_n),

    where :math:`Q_w(w_{n, m, i, j})` is the weight quantization function
    and :math:`Q_b(b_{n})` is the bias quantization function.
    """

    def __init__(self, in_channel: int, out_channel: int, **conv_kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, **conv_kwargs)
        self.qw, self.qb = get_weight_quantizers(self.conv.weight, self.conv.bias)

    def forward(self, x):
        # TODO: modify the weight and bias of the conv layer in place or not?
        w = self.qw(self.conv.weight)
        b = None if self.qb is None else self.qb(self.conv.bias)
        return self.conv._conv_forward(x, w, b)

    def get_weight_bytes(self):
        # self.qw.b is differentiable against the quantizer parameters,
        # while self.conv.weight.numel() is a constant.
        weight = self.qw.b() * self.conv.weight.numel()
        bias = 0 if self.qb is None else self.qb.b() * self.conv.bias.numel()  # type: ignore
        return (weight + bias) / 8.0

    def get_activ_bytes(self):
        return torch.tensor(0.0)


class MPQLinear(MPQModule):
    r"""Quantized Affine.

    Quantized affine with

    .. math::

        y_j = \sum_{i} Q_w(w_{ji}) x_i + Q_b(b_j),

    where :math:`Q_w(.)` is the weight quantization function
    and :math:`Q_b(.)` the bias quantization function, respectively.
    """

    def __init__(self, in_features: int, out_features: int, **linear_kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, **linear_kwargs)
        self.qw, self.qb = get_weight_quantizers(self.linear.weight, self.linear.bias)

    def forward(self, x):
        w = self.qw(self.linear.weight)
        b = None if self.qb is None else self.qb(self.linear.bias)
        return nn.functional.linear(x, w, b)

    def get_weight_bytes(self):
        weight = self.qw.b() * self.linear.weight.numel()
        bias = 0 if self.qb is None else self.qb.b() * self.linear.bias.numel()
        return (weight + bias) / 8.0

    def get_activ_bytes(self):
        return torch.tensor(0.0)


class MPQReLU(MPQModule):
    def __init__(self) -> None:
        super().__init__()
        step_init = 2**-3
        activ_bitwidth = 8
        qmax_init = step_init * (2.0**activ_bitwidth - 1)
        step = QuantParamRange(2**-3, 2**-8, 1)
        qmax = QuantParamRange(qmax_init, 2**-8, 255)
        self.quant_a = Quantizer(qmax, step, signed=False)
        self._last_activ_shape: torch.Size | None = None

    def forward(self, x: Tensor) -> Tensor:
        ret = self.quant_a(x)
        if self.quant_a.activated and not self.quant_a.signed:
            # No need to apply ReLU -- quantization is unsigned
            assert torch.all(ret >= 0)
        else:
            # we don't apply quantization, so we need to apply ReLU here
            ret = torch.relu(ret)
        assert ret.shape == x.shape
        self._last_activ_shape = x.shape
        return ret

    def get_weight_bytes(self):
        return torch.tensor(0.0)

    def get_activ_bytes(self):
        assert self._last_activ_shape is not None
        return self.quant_a.b() * np.prod(self._last_activ_shape) / 8.0
