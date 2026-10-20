"""
Spiking Neural Network (SNN) Core Components.

Implements:
1. Surrogate gradient functions (FastSigmoid, ATan, PiecewiseQuadratic) with robust tensor-shape autograd.
2. SNN Neuron Models:
   - LIFNeuron: Standard Leaky Integrate-and-Fire.
   - PLIFNeuron: Parametric LIF with learnable per-channel decay and threshold.
   - ALIFNeuron: Adaptive LIF with spike-frequency threshold adaptation.
   - APLIFNeuron: Combined Adaptive Parametric LIF.
   - FSNeuron: Few-Spike / Multi-Threshold Bit-Plane Neuron.
3. SpikingConvBlock1d: Dilated 1-D Depthwise-Separable Spiking Convolution Block.

Note on reset detach:
    The spike tensor from the previous step is DETACHED before it is used in the
    subtract-reset equation:  u[t] = β·u[t-1] - s[t-1].detach()·V_th + (1-β)·I[t]
    This is the standard BPTT-through-surrogates formulation (also used by snntorch
    and SpikingJelly). Without detach(), gradients fold back through the previous
    step's spike decision, causing gradient instability.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union, List


def _reduce_grad(grad: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Reduces grad tensor to match the target tensor shape across broadcasted dimensions."""
    if grad.shape == target.shape:
        return grad
    g = grad
    while g.dim() > target.dim():
        g = g.sum(0)
    for dim, (s_g, s_t) in enumerate(zip(g.shape, target.shape)):
        if s_t == 1 and s_g > 1:
            g = g.sum(dim, keepdim=True)
    return g


# ==============================================================================
# 1. Surrogate Gradient Functions
# ==============================================================================

class FastSigmoidSurrogate(torch.autograd.Function):
    """
    Fast Sigmoid surrogate gradient for non-differentiable threshold spike function.
    Forward:
        S = 1.0 if U >= V_th else 0.0
    Backward:
        dS/dU ~= 1 / (1 + slope * |U - V_th|)^2
    """
    @staticmethod
    def forward(ctx, mem: torch.Tensor, threshold: Union[float, torch.Tensor] = 1.0, slope: float = 10.0):
        thresh_is_tensor = isinstance(threshold, torch.Tensor)
        if thresh_is_tensor:
            ctx.save_for_backward(mem, threshold)
        else:
            ctx.save_for_backward(mem)
        ctx.thresh_is_tensor = thresh_is_tensor
        ctx.threshold_val = float(threshold) if not thresh_is_tensor else 0.0
        ctx.slope = slope
        spike = (mem >= threshold).to(mem.dtype)
        return spike

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.thresh_is_tensor:
            mem, threshold = ctx.saved_tensors
        else:
            mem, = ctx.saved_tensors
            threshold = ctx.threshold_val
        slope = ctx.slope
        
        # Calculate in float32 to prevent FP16 squaring overflow (diff >= 26 would overflow FP16 to inf)
        mem_f = mem.float()
        thresh_f = threshold.float() if ctx.thresh_is_tensor else threshold
        diff = torch.abs(mem_f - thresh_f)
        grad_input = (grad_output.float() / (1.0 + slope * diff) ** 2).to(mem.dtype)
        grad_thresh = None
        if ctx.needs_input_grad[1] and ctx.thresh_is_tensor:
            grad_thresh = _reduce_grad(-grad_input, threshold)
        return grad_input, grad_thresh, None


class ATanSurrogate(torch.autograd.Function):
    """
    ArcTangent (ATan) surrogate gradient.
    Backward:
        dS/dU ~= (alpha / 2) / (1 + (pi/2 * alpha * (U - V_th))^2)
    """
    @staticmethod
    def forward(ctx, mem: torch.Tensor, threshold: Union[float, torch.Tensor] = 1.0, alpha: float = 2.0):
        thresh_is_tensor = isinstance(threshold, torch.Tensor)
        if thresh_is_tensor:
            ctx.save_for_backward(mem, threshold)
        else:
            ctx.save_for_backward(mem)
        ctx.thresh_is_tensor = thresh_is_tensor
        ctx.threshold_val = float(threshold) if not thresh_is_tensor else 0.0
        ctx.alpha = alpha
        spike = (mem >= threshold).to(mem.dtype)
        return spike

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.thresh_is_tensor:
            mem, threshold = ctx.saved_tensors
        else:
            mem, = ctx.saved_tensors
            threshold = ctx.threshold_val
        alpha = ctx.alpha
        
        # Calculate in float32 to prevent FP16 squaring overflow
        mem_f = mem.float()
        thresh_f = threshold.float() if ctx.thresh_is_tensor else threshold
        delta = mem_f - thresh_f
        denom = 1.0 + (math.pi / 2.0 * alpha * delta) ** 2
        grad_input = (grad_output.float() * (alpha / 2.0) / denom).to(mem.dtype)
        grad_thresh = None
        if ctx.needs_input_grad[1] and ctx.thresh_is_tensor:
            grad_thresh = _reduce_grad(-grad_input, threshold)
        return grad_input, grad_thresh, None


class PiecewiseQuadraticSurrogate(torch.autograd.Function):
    """
    Piecewise Quadratic surrogate gradient.
    """
    @staticmethod
    def forward(ctx, mem: torch.Tensor, threshold: Union[float, torch.Tensor] = 1.0, width: float = 1.0):
        thresh_is_tensor = isinstance(threshold, torch.Tensor)
        if thresh_is_tensor:
            ctx.save_for_backward(mem, threshold)
        else:
            ctx.save_for_backward(mem)
        ctx.thresh_is_tensor = thresh_is_tensor
        ctx.threshold_val = float(threshold) if not thresh_is_tensor else 0.0
        ctx.width = width
        spike = (mem >= threshold).to(mem.dtype)
        return spike

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.thresh_is_tensor:
            mem, threshold = ctx.saved_tensors
        else:
            mem, = ctx.saved_tensors
            threshold = ctx.threshold_val
        width = ctx.width
        
        # Calculate in float32 for numerical stability
        mem_f = mem.float()
        thresh_f = threshold.float() if ctx.thresh_is_tensor else threshold
        diff = mem_f - thresh_f
        abs_diff = torch.abs(diff)
        mask = (abs_diff <= width).float()
        denom = max(width ** 2, 1e-8)
        grad_input = (grad_output.float() * mask * (width - abs_diff) / denom).to(mem.dtype)
        grad_thresh = None
        if ctx.needs_input_grad[1] and ctx.thresh_is_tensor:
            grad_thresh = _reduce_grad(-grad_input, threshold)
        return grad_input, grad_thresh, None



def get_surrogate_fn(surrogate_type: str = "fast_sigmoid"):
    """Returns the surrogate spike function based on configuration."""
    surr = surrogate_type.lower()
    if surr == "atan":
        return ATanSurrogate.apply
    elif surr in ["piecewise", "quadratic"]:
        return PiecewiseQuadraticSurrogate.apply
    return FastSigmoidSurrogate.apply


# ==============================================================================
# 2. SNN Neuron Models
# ==============================================================================

class LIFNeuron(nn.Module):
    """
    Standard Leaky Integrate-and-Fire (LIF) Neuron layer.
    Discrete dynamics:
        U[t] = beta * U[t-1] - S[t-1] * V_th + (1 - beta) * I[t]
        S[t] = SurrogateSpike(U[t] - V_th)
    """

    def __init__(
        self,
        channels: Optional[int] = None,
        threshold: float = 0.8,
        beta: float = 0.9,
        reset_mechanism: str = "subtract",
        surrogate: str = "fast_sigmoid",
    ):
        super().__init__()
        self.threshold = threshold
        self.beta = beta
        self.reset_mechanism = reset_mechanism
        self.spike_fn = get_surrogate_fn(surrogate)

    def init_state(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros_like(x), torch.zeros_like(x)

    def step(
        self,
        input_current: torch.Tensor,
        u: torch.Tensor,
        s: torch.Tensor,
        step_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        s_d = s.detach()  # Detach reset spike — gradient flows only through threshold crossing
        if self.reset_mechanism == "zero":
            u_decayed = self.beta * u * (1.0 - s_d) + input_current
        else:
            u_decayed = self.beta * u - s_d * self.threshold + input_current

        s_new = self.spike_fn(u_decayed, self.threshold)
        return s_new, u_decayed


class PLIFNeuron(nn.Module):
    """
    Parametric Leaky Integrate-and-Fire (PLIF) Neuron.
    Learnable per-channel decay parameter:
        beta_c = sigmoid(w_beta_c)
    Learnable per-channel firing threshold:
        V_th_c = softplus(w_th_c)
    """

    def __init__(
        self,
        channels: int = 64,
        init_beta: float = 0.9,
        init_threshold: float = 0.8,
        reset_mechanism: str = "subtract",
        surrogate: str = "fast_sigmoid",
    ):
        super().__init__()
        self.channels = channels
        self.reset_mechanism = reset_mechanism
        self.spike_fn = get_surrogate_fn(surrogate)

        init_beta = max(min(init_beta, 0.999), 0.001)
        w_beta_val = math.log(init_beta / (1.0 - init_beta))
        self.w_beta = nn.Parameter(torch.full((1, channels, 1), w_beta_val, dtype=torch.float32))

        init_thresh_clamped = max(float(init_threshold), 1e-4)
        w_th_val = math.log(math.expm1(init_thresh_clamped)) if init_thresh_clamped < 20.0 else init_thresh_clamped
        self.w_th = nn.Parameter(torch.full((1, channels, 1), w_th_val, dtype=torch.float32))

    @property
    def beta(self) -> torch.Tensor:
        return torch.sigmoid(self.w_beta)

    @property
    def threshold(self) -> torch.Tensor:
        return F.softplus(self.w_th) + 1e-4

    def init_state(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros_like(x), torch.zeros_like(x)

    def step(
        self,
        input_current: torch.Tensor,
        u: torch.Tensor,
        s: torch.Tensor,
        step_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        beta = self.beta
        v_th = self.threshold
        s_d = s.detach()  # Detach reset spike — gradient flows only through threshold crossing

        if self.reset_mechanism == "zero":
            u_decayed = beta * u * (1.0 - s_d) + input_current
        else:
            u_decayed = beta * u - s_d * v_th + input_current

        s_new = self.spike_fn(u_decayed, v_th)
        return s_new, u_decayed


class ALIFNeuron(nn.Module):
    """
    Adaptive Leaky Integrate-and-Fire (ALIF) Neuron with dynamic threshold adaptation.
    Threshold adaptation:
        a[t] = rho * a[t-1] + S[t-1]
        V_th[t] = V_0 + beta_ada * a[t]
    """

    def __init__(
        self,
        channels: int = 64,
        threshold: float = 0.8,
        beta: float = 0.9,
        rho: float = 0.85,
        beta_ada: float = 0.2,
        reset_mechanism: str = "subtract",
        surrogate: str = "fast_sigmoid",
    ):
        super().__init__()
        self.channels = channels
        self.base_threshold = threshold
        self.beta = beta
        self.rho = rho
        self.beta_ada = beta_ada
        self.reset_mechanism = reset_mechanism
        self.spike_fn = get_surrogate_fn(surrogate)

    def init_state(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u = torch.zeros_like(x)
        s = torch.zeros_like(x)
        a = torch.zeros_like(x)
        return u, s, a

    def step(
        self,
        input_current: torch.Tensor,
        u: torch.Tensor,
        s: torch.Tensor,
        a: Optional[torch.Tensor] = None,
        step_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if a is None:
            a = torch.zeros_like(u)

        s_d = s.detach()  # Detach reset spike — gradient flows only through threshold crossing
        a_new = self.rho * a + s_d
        v_th = self.base_threshold + self.beta_ada * a_new

        if self.reset_mechanism == "zero":
            u_decayed = self.beta * u * (1.0 - s_d) + input_current
        else:
            u_decayed = self.beta * u - s_d * v_th + input_current

        s_new = self.spike_fn(u_decayed, v_th)
        return s_new, u_decayed, a_new


class APLIFNeuron(nn.Module):
    """
    Combined Adaptive-Parametric LIF (APLIF) Neuron.
    Combines learnable channel-wise decay beta and learnable baseline threshold
    with dynamic threshold adaptation.
    """

    def __init__(
        self,
        channels: int = 64,
        init_beta: float = 0.9,
        init_threshold: float = 0.8,
        init_rho: float = 0.85,
        init_beta_ada: float = 0.2,
        reset_mechanism: str = "subtract",
        surrogate: str = "fast_sigmoid",
    ):
        super().__init__()
        self.channels = channels
        self.reset_mechanism = reset_mechanism
        self.spike_fn = get_surrogate_fn(surrogate)

        init_beta = max(min(init_beta, 0.999), 0.001)
        w_beta_val = math.log(init_beta / (1.0 - init_beta))
        self.w_beta = nn.Parameter(torch.full((1, channels, 1), w_beta_val, dtype=torch.float32))

        init_thresh_clamped = max(float(init_threshold), 1e-4)
        w_th_val = math.log(math.expm1(init_thresh_clamped)) if init_thresh_clamped < 20.0 else init_thresh_clamped
        self.w_th = nn.Parameter(torch.full((1, channels, 1), w_th_val, dtype=torch.float32))

        self.w_ada = nn.Parameter(torch.full((1, channels, 1), init_beta_ada, dtype=torch.float32))
        self.rho = init_rho

    @property
    def beta(self) -> torch.Tensor:
        return torch.sigmoid(self.w_beta)

    @property
    def base_threshold(self) -> torch.Tensor:
        return F.softplus(self.w_th) + 1e-4

    @property
    def beta_ada(self) -> torch.Tensor:
        return F.softplus(self.w_ada)

    def init_state(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)

    def step(
        self,
        input_current: torch.Tensor,
        u: torch.Tensor,
        s: torch.Tensor,
        a: Optional[torch.Tensor] = None,
        step_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if a is None:
            a = torch.zeros_like(u)

        s_d = s.detach()  # Detach reset spike — gradient flows only through threshold crossing
        a_new = self.rho * a + s_d
        v_th = self.base_threshold + self.beta_ada * a_new
        beta = self.beta

        if self.reset_mechanism == "zero":
            u_decayed = beta * u * (1.0 - s_d) + input_current
        else:
            u_decayed = beta * u - s_d * v_th + input_current

        s_new = self.spike_fn(u_decayed, v_th)
        return s_new, u_decayed, a_new


class FSNeuron(nn.Module):
    """
    Few-Spike / Multi-Threshold Bit-Plane Neuron (FS-Neuron).
    Each simulation timestep t represents a base-2 positional radix threshold:
        V_th[t] = V_0 * 2^(-t) (or learnable bit weights)
    Enables exact representation of 2^S discrete amplitude levels in S timesteps.
    """

    def __init__(
        self,
        channels: int = 64,
        timesteps: int = 4,
        base_threshold: float = 0.8,
        learnable_weights: bool = True,
        surrogate: str = "fast_sigmoid",
    ):
        super().__init__()
        self.channels = channels
        self.timesteps = timesteps
        self.spike_fn = get_surrogate_fn(surrogate)

        powers = torch.tensor([2.0 ** (-t) for t in range(timesteps)], dtype=torch.float32)
        if learnable_weights:
            self.bit_scale = nn.Parameter(powers.view(timesteps, 1, 1, 1))
            self.base_vth = nn.Parameter(torch.full((1, channels, 1), base_threshold, dtype=torch.float32))
        else:
            self.register_buffer("bit_scale", powers.view(timesteps, 1, 1, 1))
            self.register_buffer("base_vth", torch.full((1, channels, 1), base_threshold, dtype=torch.float32))

    def get_threshold(self, step_idx: int) -> torch.Tensor:
        idx = max(0, min(step_idx, self.timesteps - 1))
        scale = torch.abs(self.bit_scale[idx]) + 1e-6
        vth = F.softplus(self.base_vth) + 1e-4 if isinstance(self.base_vth, nn.Parameter) else self.base_vth
        return vth * scale

    def init_state(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros_like(x), torch.zeros_like(x)

    def step(
        self,
        input_current: torch.Tensor,
        u: torch.Tensor,
        s: torch.Tensor,
        step_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        s_d = s.detach()  # Detach reset spike — gradient flows only through threshold crossing
        v_th_reset = self.get_threshold(step_idx - 1) if step_idx > 0 else self.get_threshold(0)
        v_th = self.get_threshold(step_idx)
        u_decayed = u - s_d * v_th_reset + input_current
        s_new = self.spike_fn(u_decayed, v_th)
        return s_new, u_decayed


def get_neuron(
    neuron_type: str = "plif",
    channels: int = 64,
    threshold: float = 0.8,
    beta: float = 0.9,
    timesteps: int = 4,
    reset_mechanism: str = "subtract",
    surrogate: str = "fast_sigmoid",
) -> nn.Module:
    """Factory function for instantiating SNN neuron models."""
    n_type = neuron_type.lower()
    if n_type == "plif":
        return PLIFNeuron(
            channels=channels,
            init_beta=beta,
            init_threshold=threshold,
            reset_mechanism=reset_mechanism,
            surrogate=surrogate,
        )
    elif n_type == "alif":
        return ALIFNeuron(
            channels=channels,
            threshold=threshold,
            beta=beta,
            reset_mechanism=reset_mechanism,
            surrogate=surrogate,
        )
    elif n_type in ["aplif", "adaptive_plif"]:
        return APLIFNeuron(
            channels=channels,
            init_beta=beta,
            init_threshold=threshold,
            reset_mechanism=reset_mechanism,
            surrogate=surrogate,
        )
    elif n_type in ["fs_neuron", "fsneuron", "bit_plane"]:
        return FSNeuron(
            channels=channels,
            timesteps=timesteps,
            base_threshold=threshold,
            surrogate=surrogate,
        )
    else:
        return LIFNeuron(
            channels=channels,
            threshold=threshold,
            beta=beta,
            reset_mechanism=reset_mechanism,
            surrogate=surrogate,
        )


# ==============================================================================
# 3. Spiking Dilated 1-D Convolution Block
# ==============================================================================

class SpikingConvBlock1d(nn.Module):
    """
    Dilated 1-D Depthwise-Separable Convolution Block with selectable SNN Neurons.
    Returns both residual/skip spike sequences and accumulated continuous membrane state.

    Architecture per block:
        1. 1x1 Conv (Bottleneck -> Hidden channels)
        2. Neuron 1 (LIF / PLIF / ALIF / APLIF / FSNeuron)
        3. Dilated Depthwise 1-D Conv (Hidden -> Hidden channels)
        4. Neuron 2 (LIF / PLIF / ALIF / APLIF / FSNeuron)
        5. 1x1 Conv (Hidden -> Bottleneck channels) -> Residual connection
        6. 1x1 Conv (Hidden -> Bottleneck channels) -> Skip connection
    """

    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 128,
        kernel_size: int = 3,
        dilation: int = 1,
        neuron_type: str = "plif",
        beta: float = 0.9,
        threshold: float = 0.8,
        timesteps: int = 4,
        surrogate: str = "fast_sigmoid",
        return_membrane: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dilation = dilation
        self.neuron_type = neuron_type.lower()
        self.timesteps = timesteps
        self.return_membrane = return_membrane

        # 1x1 Pointwise conv expansion: [B_conv -> H]
        self.conv1x1_in = nn.Conv1d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(1, hidden_channels, eps=1e-8)
        self.neuron1 = get_neuron(
            neuron_type=neuron_type,
            channels=hidden_channels,
            threshold=threshold,
            beta=beta,
            timesteps=timesteps,
            surrogate=surrogate,
        )

        # Depthwise dilated 1-D conv: [H -> H]
        padding = (kernel_size - 1) * dilation // 2
        self.depthwise_conv = nn.Conv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            groups=hidden_channels,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(1, hidden_channels, eps=1e-8)
        self.neuron2 = get_neuron(
            neuron_type=neuron_type,
            channels=hidden_channels,
            threshold=threshold,
            beta=beta,
            timesteps=timesteps,
            surrogate=surrogate,
        )

        # 1x1 Pointwise conv projection: [H -> B_conv]
        self.conv1x1_res = nn.Conv1d(hidden_channels, in_channels, kernel_size=1, bias=False)
        self.conv1x1_skip = nn.Conv1d(hidden_channels, in_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(1, in_channels, eps=1e-8)

    def forward(
        self,
        spike_seq: torch.Tensor,
        return_spikes: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]]:
        """
        Process SNN temporal sequence across all simulation timesteps.

        Args:
            spike_seq: Input sequence of shape [Timesteps S, Batch, In_Channels, Latent_Length].
            return_spikes: If True, also return s2_seq spike tensor.

        Returns:
            Tuple (res_seq, skip_seq, u2_seq) or (res_seq, skip_seq, u2_seq, s2_seq):
                - res_seq:  Residual output spikes [S, B, In_Channels, L]
                - skip_seq: Skip output spikes [S, B, In_Channels, L]
                - u2_seq:   Membrane potential sequence of hidden layer [S, B, In_Channels, L]
        """
        s_steps, batch, in_c, l_len = spike_seq.shape
        is_adaptive = any(x in self.neuron_type for x in ["alif", "aplif"])

        # Step 1: Vectorized 1x1 input expansion for all S simulation steps
        flat_x = spike_seq.reshape(s_steps * batch, in_c, l_len).to(dtype=self.conv1x1_in.weight.dtype)
        flat_h1 = self.norm1(self.conv1x1_in(flat_x))
        h1_seq = flat_h1.reshape(s_steps, batch, self.hidden_channels, l_len)

        dtype = flat_h1.dtype
        # Step 2: Temporal recurrent SNN simulation
        u1 = torch.zeros(batch, self.hidden_channels, l_len, dtype=dtype, device=spike_seq.device)
        s1 = torch.zeros(batch, self.hidden_channels, l_len, dtype=dtype, device=spike_seq.device)
        u2 = torch.zeros(batch, self.hidden_channels, l_len, dtype=dtype, device=spike_seq.device)
        s2 = torch.zeros(batch, self.hidden_channels, l_len, dtype=dtype, device=spike_seq.device)
        
        a1 = torch.zeros(batch, self.hidden_channels, l_len, dtype=dtype, device=spike_seq.device) if is_adaptive else None
        a2 = torch.zeros(batch, self.hidden_channels, l_len, dtype=dtype, device=spike_seq.device) if is_adaptive else None

        s2_list = []
        u2_list = []

        for t in range(s_steps):
            # Neuron 1 step
            if is_adaptive:
                s1, u1, a1 = self.neuron1.step(h1_seq[t], u1, s1, a=a1, step_idx=t)
            else:
                s1, u1 = self.neuron1.step(h1_seq[t], u1, s1, step_idx=t)

            # Depthwise conv
            s1_in = s1.to(dtype=self.depthwise_conv.weight.dtype)
            h2 = self.norm2(self.depthwise_conv(s1_in))

            # Neuron 2 step
            if is_adaptive:
                s2, u2, a2 = self.neuron2.step(h2, u2, s2, a=a2, step_idx=t)
            else:
                s2, u2 = self.neuron2.step(h2, u2, s2, step_idx=t)

            s2_list.append(s2)
            u2_list.append(u2)

        # Step 3: Vectorized residual and skip projections
        s2_seq = torch.stack(s2_list, dim=0)
        u2_raw_seq = torch.stack(u2_list, dim=0)
        self.last_spikes = s2_seq.detach()

        flat_s2 = s2_seq.reshape(s_steps * batch, self.hidden_channels, l_len).to(dtype=self.conv1x1_res.weight.dtype)
        flat_res = self.conv1x1_res(flat_s2)
        flat_skip = self.conv1x1_skip(flat_s2)

        flat_res_out = self.norm(flat_res + flat_x)

        res_seq = flat_res_out.reshape(s_steps, batch, in_c, l_len)
        skip_seq = flat_skip.reshape(s_steps, batch, in_c, l_len)

        if self.return_membrane:
            flat_u2 = u2_raw_seq.reshape(s_steps * batch, self.hidden_channels, l_len).to(dtype=self.conv1x1_skip.weight.dtype)
            # Reuse skip connection projection (conv1x1_skip) for membrane readout
            flat_u_proj = self.conv1x1_skip(flat_u2)
            u_seq = flat_u_proj.reshape(s_steps, batch, in_c, l_len)
        else:
            u_seq = None

        if return_spikes:
            return res_seq, skip_seq, u_seq, s2_seq
        return res_seq, skip_seq, u_seq
