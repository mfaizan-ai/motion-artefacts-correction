#!/usr/bin/env python3
"""
building_blocks.py
==================

a torch module containing building blocks for the model, such as convolutional layers, activation functions, and normalization layers.
"""
import torch
import torch.nn as nn

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import torch
from torch import Tensor, nn

Kernel3D: TypeAlias = int | tuple[int, int, int]
State3D: TypeAlias = tuple[Tensor, Tensor]



# 3D Residual Block
class ResBlock3D(nn.Module):
    """
    3D Residual block with Instance Normalisation.

    Used in the ContentEncoder bottleneck.
    Preserves feature map dimensions — no spatial change.

    Input / Output: (B, channels, D, H, W)
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = False,
        negative_slope: float = 0.2,
        affine: bool = True,
        inplace: bool = True,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            ),
            nn.InstanceNorm3d(channels, affine=affine),
            nn.LeakyReLU(negative_slope, inplace=inplace),
            nn.Conv3d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            ),
            nn.InstanceNorm3d(channels, affine=affine),
        )

        self.activation = nn.LeakyReLU(negative_slope, inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))
    
# 3D Strided Convolution Block 
class StridedConvBlock(nn.Module):
    """
    3D strided convolution block for spatial downsampling.

    Each block halves spatial dimensions:
        (D, H, W) → (D/2, H/2, W/2)

    Parameters
    ----------
    in_ch       : int   input channels
    out_ch      : int   output channels
    use_norm    : bool  if True uses InstanceNorm (content encoder)
                        if False uses BatchNorm (artefact encoder)
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        use_norm: bool = True,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        instance_affine: bool = True,
        negative_slope: float = 0.2,
        inplace: bool = True,
    ):
        super().__init__()

        layers = [
            nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=not use_norm,
            ),
        ]

        if use_norm:
            # InstanceNorm: normalises per sample per channel
            # Removes mean/variance info → encourages content-only encoding
            layers.append(nn.InstanceNorm3d(out_ch, affine=instance_affine))
        else:
            # BatchNorm: preserves cross-sample statistics
            # Keeps mean/variance info → artefact encoder retains
            # intensity shift and variance spike signatures of motion
            layers.append(nn.BatchNorm3d(out_ch))

        layers.append(nn.LeakyReLU(negative_slope, inplace=inplace))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
    
    
class UpBlock3D(nn.Module):
    """
    3D upsampling block: interpolation + Conv + InstanceNorm + LReLU.

    Each block increases spatial dimensions according to scale_factor:
        (D, H, W) → (scale_factor*D, scale_factor*H, scale_factor*W)

    Input / Output channels are configurable.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        scale_factor: int = 2,
        mode: str = "trilinear",
        align_corners: bool = False,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = False,
        affine: bool = True,
        negative_slope: float = 0.2,
        inplace: bool = True,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Upsample(
                scale_factor=scale_factor,
                mode=mode,
                align_corners=align_corners,
            ),
            nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            ),
            nn.InstanceNorm3d(out_ch, affine=affine),
            nn.LeakyReLU(negative_slope, inplace=inplace),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
    
    
class AdaIN3D(nn.Module):
    """
    Adaptive Instance Normalisation for 3D feature maps.

    Replaces fixed IN affine parameters with artefact-code-conditioned ones.

        AdaIN(x, a) = σ(a) · InstanceNorm(x) + μ(a)

    Parameters
    ----------
    channels      : int   number of feature map channels to modulate
    artefact_dim  : int   dimension of the global artefact code
    """

    def __init__(
        self,
        channels: int,
        artefact_dim: int = 64,
        affine: bool = False,
        bias: bool = True,
        mean_init: float | None = None,
        std_init: float | None = None,
    ):
        super().__init__()

        self.norm = nn.InstanceNorm3d(channels, affine=affine)
        self.mlp_mean = nn.Linear(artefact_dim, channels, bias=bias)
        self.mlp_std = nn.Linear(artefact_dim, channels, bias=bias)

        if mean_init is not None:
            nn.init.constant_(self.mlp_mean.bias, mean_init)

        if std_init is not None:
            nn.init.constant_(self.mlp_std.bias, std_init)

    def forward(
        self,
        x: torch.Tensor,
        a_global: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x        : (B, C, D, H, W)   feature map to modulate
        a_global : (B, artefact_dim) global artefact code

        Returns
        -------
        (B, C, D, H, W)  modulated feature map
        """
        mean = self.mlp_mean(a_global)
        std = self.mlp_std(a_global)

        mean = mean[:, :, None, None, None]
        std = std[:, :, None, None, None]

        return std * self.norm(x) + mean
    
    
class AdaINResBlock3D(nn.Module):
    """
    3D residual block where both IN layers are replaced with AdaIN.

    Each residual block applies two AdaIN operations, both conditioned
    on the same global artefact code.

    Input / Output: (B, channels, D, H, W)
    """

    def __init__(
        self,
        channels: int,
        artefact_dim: int = 64,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = False,
        negative_slope: float = 0.2,
        inplace: bool = True,
        adain_affine: bool = False,
        adain_bias: bool = True,
        mean_init: float | None = None,
        std_init: float | None = None,
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )
        self.adain1 = AdaIN3D(
            channels,
            artefact_dim=artefact_dim,
            affine=adain_affine,
            bias=adain_bias,
            mean_init=mean_init,
            std_init=std_init,
        )
        self.act1 = nn.LeakyReLU(negative_slope, inplace=inplace)

        self.conv2 = nn.Conv3d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )
        self.adain2 = AdaIN3D(
            channels,
            artefact_dim=artefact_dim,
            affine=adain_affine,
            bias=adain_bias,
            mean_init=mean_init,
            std_init=std_init,
        )
        self.act2 = nn.LeakyReLU(negative_slope, inplace=inplace)

    def forward(
        self,
        x: torch.Tensor,
        a_global: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x        : (B, C, D, H, W)
        a_global : (B, artefact_dim)
        """
        residual = x

        x = self.conv1(x)
        x = self.adain1(x, a_global)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.adain2(x, a_global)

        return self.act2(x + residual)
    

def compute_temporal_diffs(x: torch.Tensor) -> torch.Tensor:
    """
    Compute frame-to-frame temporal difference maps.
 
    For a chunk with T timepoints, produces T-1 difference maps:
        Δx[t] = x[t+1] - x[t]   for t = 0, 1, ..., T-2
 
    These capture the rate of temporal change — motion artefacts produce
    sudden large differences while clean BOLD signal changes slowly and
    smoothly. By appending these as extra input channels the discriminator
    can directly evaluate temporal dynamics rather than only spatial patterns.
 
    Parameters
    ----------
    x : (B, T, D, H, W)   fMRI chunk, T timepoints as channels
 
    Returns
    -------
    diffs : (B, T-1, D, H, W)   frame-to-frame differences
    """
    # x[:, 1:] = volumes 1..T-1
    # x[:, :-1] = volumes 0..T-2
    # diff[t] = volume[t+1] - volume[t]
    return x[:, 1:, ...] - x[:, :-1, ...]
 
 
def append_temporal_diffs(x: torch.Tensor) -> torch.Tensor:
    """
    Append temporal difference maps to the chunk as extra channels.
 
    Concatenates the original T timepoints with T-1 difference maps,
    giving a (B, 2T-1, D, H, W) tensor as discriminator input.
 
    For T=20: input becomes (B, 39, D, H, W)
        channels  0–19  : original volumes
        channels 20–38  : frame-to-frame differences
 
    Parameters
    ----------
    x : (B, T, D, H, W)
 
    Returns
    -------
    (B, 2T-1, D, H, W)
    """
    diffs = compute_temporal_diffs(x)       # (B, T-1, D, H, W)
    return torch.cat([x, diffs], dim=1)     # (B, 2T-1, D, H, W)
 
 
class DiscConvBlock(nn.Module):
    """
    Single discriminator convolutional block.
 
    Conv3D (stride 2) + optional InstanceNorm + LeakyReLU.
    Spectral normalisation is applied to the conv weight matrix to
    constrain the Lipschitz constant of the discriminator and prevent
    it from dominating the generator during training (Miyato et al. 2018).
 
    The first block omits normalisation following the pix2pix convention —
    normalising the first layer can destabilise early training when the
    input statistics are still widely varying.
 
    Parameters
    ----------
    in_ch    : int   input channels
    out_ch   : int   output channels
    use_norm : bool  apply InstanceNorm after conv (False for first block)
    """
 
    def __init__(self, in_ch: int, out_ch: int, use_norm: bool = True):
        super().__init__()
 
        # Spectral normalisation wraps the Conv3d weight matrix
        conv = nn.utils.spectral_norm(
            nn.Conv3d(in_ch, out_ch,
                      kernel_size=4, stride=2, padding=1, bias=not use_norm)
        )
        layers = [conv]
 
        if use_norm:
            layers.append(nn.InstanceNorm3d(out_ch, affine=True))
 
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
    
    
    
def _to_triple(value: Kernel3D) -> tuple[int, int, int]:
    if isinstance(value, int):
        value = (value, value, value)
    if len(value) != 3 or any(k <= 0 for k in value):
        raise ValueError(f"kernel_size must contain three positive values, got {value}")
    if any(k % 2 == 0 for k in value):
        raise ValueError(
            "Only odd ConvLSTM kernels are supported so same-padding preserves "
            f"the spatial shape; received {value}."
        )
    return tuple(int(k) for k in value)


class ConvLSTM3DCell(nn.Module):
    """One ConvLSTM cell whose gates use a 3D convolution.

    Args:
        input_channels: Channels in the current input volume ``x_t``.
        hidden_channels: Channels in the hidden and cell states.
        kernel_size: Odd 3D kernel size. ``3`` means ``(3, 3, 3)``.
        bias: Whether the gate convolution includes a bias.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: Kernel3D = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or hidden_channels <= 0:
            raise ValueError("input_channels and hidden_channels must be positive")

        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.kernel_size = _to_triple(kernel_size)
        padding = tuple(k // 2 for k in self.kernel_size)

        # One convolution produces input, forget, output and candidate tensors.
        self.gates = nn.Conv3d(
            in_channels=self.input_channels + self.hidden_channels,
            out_channels=4 * self.hidden_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=padding,
            bias=bias,
        )

    def initial_state(self, x_t: Tensor) -> State3D:
        """Create zero states matching ``x_t`` device, dtype and spatial size."""
        if x_t.ndim != 5:
            raise ValueError(f"x_t must be [B,C,D,H,W], got {tuple(x_t.shape)}")
        batch, _, depth, height, width = x_t.shape
        shape = (batch, self.hidden_channels, depth, height, width)
        return x_t.new_zeros(shape), x_t.new_zeros(shape)

    def forward(self, x_t: Tensor, state: State3D | None = None) -> State3D:
        if x_t.ndim != 5:
            raise ValueError(f"x_t must be [B,C,D,H,W], got {tuple(x_t.shape)}")
        if x_t.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, got {x_t.shape[1]}"
            )

        if state is None:
            h_previous, c_previous = self.initial_state(x_t)
        else:
            h_previous, c_previous = state

        expected_state_shape = (
            x_t.shape[0],
            self.hidden_channels,
            x_t.shape[2],
            x_t.shape[3],
            x_t.shape[4],
        )
        if tuple(h_previous.shape) != expected_state_shape:
            raise ValueError(
                f"Hidden state must be {expected_state_shape}, got {tuple(h_previous.shape)}"
            )
        if tuple(c_previous.shape) != expected_state_shape:
            raise ValueError(
                f"Cell state must be {expected_state_shape}, got {tuple(c_previous.shape)}"
            )

        combined = torch.cat((x_t, h_previous), dim=1)
        gates = self.gates(combined)
        i_raw, f_raw, o_raw, g_raw = torch.chunk(gates, chunks=4, dim=1)

        input_gate = torch.sigmoid(i_raw)
        forget_gate = torch.sigmoid(f_raw)
        output_gate = torch.sigmoid(o_raw)
        candidate = torch.tanh(g_raw)

        c_t = forget_gate * c_previous + input_gate * candidate
        h_t = output_gate * torch.tanh(c_t)
        return h_t, c_t


class ConvLSTM3D(nn.Module):
    """A one- or multi-layer ConvLSTM for volumetric sequences.

    Default input and output layout is ``[B, T, C, D, H, W]``. When
    ``return_all_layers=False``, returns ``(last_layer_sequence, last_state)``.
    Otherwise it returns lists containing the sequence and final state of every
    recurrent layer.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int | Sequence[int],
        kernel_size: Kernel3D | Sequence[Kernel3D] = 3,
        num_layers: int = 1,
        batch_first: bool = True,
        bias: bool = True,
        return_all_layers: bool = False,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        if isinstance(hidden_channels, int):
            hidden_list = [hidden_channels] * num_layers
        else:
            hidden_list = [int(value) for value in hidden_channels]
        if len(hidden_list) != num_layers:
            raise ValueError("hidden_channels must contain one value per layer")

        if isinstance(kernel_size, int) or (
            isinstance(kernel_size, tuple)
            and len(kernel_size) == 3
            and all(isinstance(k, int) for k in kernel_size)
        ):
            kernel_list = [_to_triple(kernel_size)] * num_layers
        else:
            kernel_list = [_to_triple(value) for value in kernel_size]
        if len(kernel_list) != num_layers:
            raise ValueError("kernel_size must contain one value per layer")

        self.input_channels = int(input_channels)
        self.hidden_channels = hidden_list
        self.num_layers = int(num_layers)
        self.batch_first = bool(batch_first)
        self.return_all_layers = bool(return_all_layers)

        cells: list[ConvLSTM3DCell] = []
        for layer_index in range(num_layers):
            layer_input_channels = (
                self.input_channels
                if layer_index == 0
                else hidden_list[layer_index - 1]
            )
            cells.append(
                ConvLSTM3DCell(
                    input_channels=layer_input_channels,
                    hidden_channels=hidden_list[layer_index],
                    kernel_size=kernel_list[layer_index],
                    bias=bias,
                )
            )
        self.cells = nn.ModuleList(cells)

    def forward(
        self,
        x: Tensor,
        hidden_state: State3D | Sequence[State3D] | None = None,
    ) -> tuple[Tensor, State3D] | tuple[list[Tensor], list[State3D]]:
        if x.ndim != 6:
            layout = "[B,T,C,D,H,W]" if self.batch_first else "[T,B,C,D,H,W]"
            raise ValueError(f"Expected {layout}, got {tuple(x.shape)}")

        if not self.batch_first:
            x = x.permute(1, 0, 2, 3, 4, 5).contiguous()
        if x.shape[2] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, got {x.shape[2]}"
            )

        if hidden_state is None:
            layer_states: list[State3D | None] = [None] * self.num_layers
        elif self.num_layers == 1 and isinstance(hidden_state, tuple):
            layer_states = [hidden_state]
        else:
            layer_states = list(hidden_state)  # type: ignore[arg-type]
            if len(layer_states) != self.num_layers:
                raise ValueError("hidden_state must contain one (h, c) tuple per layer")

        current_sequence = x
        layer_outputs: list[Tensor] = []
        final_states: list[State3D] = []

        for layer_index, cell in enumerate(self.cells):
            state = layer_states[layer_index]
            outputs = []
            for time_index in range(current_sequence.shape[1]):
                state = cell(current_sequence[:, time_index], state)
                outputs.append(state[0])

            current_sequence = torch.stack(outputs, dim=1)
            layer_outputs.append(current_sequence)
            final_states.append(state)

        if not self.batch_first:
            layer_outputs = [
                output.permute(1, 0, 2, 3, 4, 5).contiguous()
                for output in layer_outputs
            ]

        if self.return_all_layers:
            return layer_outputs, final_states
        return layer_outputs[-1], final_states[-1]


class ConvLSTM3DBottleneck(nn.Module):
    """Drop-in temporal refinement for ``[B,T,C,D,H,W]`` bottlenecks.

    The recurrent result is projected back to ``channels`` with a 1x1x1
    convolution. With ``residual=True`` the returned value is
    ``x + temporal_scale * projected_recurrent_features``, so its shape is
    identical to the input and the existing decoder interface can be retained.
    """

    def __init__(
        self,
        channels: int = 384,
        hidden_channels: int = 96,
        kernel_size: Kernel3D = 3,
        bidirectional: bool = False,
        residual: bool = True,
        initial_residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.bidirectional = bool(bidirectional)
        self.residual = bool(residual)

        self.forward_lstm = ConvLSTM3D(
            input_channels=channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            batch_first=True,
        )
        if self.bidirectional:
            self.backward_lstm: ConvLSTM3D | None = ConvLSTM3D(
                input_channels=channels,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                batch_first=True,
            )
        else:
            self.backward_lstm = None

        recurrent_channels = hidden_channels * (2 if bidirectional else 1)
        self.output_projection = nn.Conv3d(
            in_channels=recurrent_channels,
            out_channels=channels,
            kernel_size=1,
        )
        self.temporal_scale = nn.Parameter(
            torch.tensor(float(initial_residual_scale))
        )

    def forward(
        self,
        x: Tensor,
        return_states: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, State3D]]:
        if x.ndim != 6:
            raise ValueError(f"Expected [B,T,C,D,H,W], got {tuple(x.shape)}")
        if x.shape[2] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[2]}")

        forward_features, forward_state = self.forward_lstm(x)
        feature_sequences = [forward_features]
        states = {"forward": forward_state}

        if self.backward_lstm is not None:
            reversed_input = torch.flip(x, dims=(1,))
            backward_reversed, backward_state = self.backward_lstm(reversed_input)
            backward_features = torch.flip(backward_reversed, dims=(1,))
            feature_sequences.append(backward_features)
            states["backward"] = backward_state

        temporal = torch.cat(feature_sequences, dim=2)
        batch, time, channels, depth, height, width = temporal.shape
        temporal = temporal.reshape(
            batch * time, channels, depth, height, width
        )
        temporal = self.output_projection(temporal)
        temporal = temporal.reshape(
            batch, time, self.channels, depth, height, width
        )

        output = x + self.temporal_scale * temporal if self.residual else temporal
        if return_states:
            return output, states
        return output


    
