from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import BaseNet

ModuleType = Type[nn.Module]


class HyperLayer(nn.Module):
    def __init__(
        self,
        noise_dim: int,
        hidden_dim: int,
        action_dim: int,
        prior_std: float = 1.0,
        trainable: bool = True,
        weight_init: str = "xavier_normal",
        bias_init: str = "default",
        device: Union[str, int, torch.device] = "cpu",
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.prior_std = prior_std
        self.trainable = trainable
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.device = device

        self.in_features = noise_dim
        self.out_features = action_dim * hidden_dim

        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.Tensor(self.out_features))
        self.reset_parameters()

        if not self.trainable:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def reset_parameters(self) -> None:
        # init weight
        if self.weight_init == "sDB":
            weight = np.random.randn(self.out_features, self.in_features).astype(np.float32)
            weight = weight / np.linalg.norm(weight, axis=1, keepdims=True)
            self.weight = nn.Parameter(torch.from_numpy(self.prior_std * weight).float())
        elif self.weight_init == "gDB":
            weight = np.random.randn(self.out_features, self.in_features).astype(np.float32)
            self.weight = nn.Parameter(torch.from_numpy(self.prior_std * weight).float())
        elif self.weight_init == "xavier_normal":
            nn.init.xavier_normal_(self.weight, gain=1.0)
        else:
            nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        # init bias
        if self.bias_init == "sphere":
            bias = np.random.randn(self.out_features).astype(np.float32)
            bias = bias / np.linalg.norm(bias)
            self.bias = nn.Parameter(torch.from_numpy(self.prior_std * bias).float())
        elif self.bias_init == "xavier_normal":
            bias = nn.init.xavier_normal_(torch.zeros((self.action_dim, self.hidden_dim)))
            self.bias = nn.Parameter(bias.flatten())
        else:
            nn.init.zeros_(self.bias)

    def forward(self, z: torch.Tensor):
        z = z.to(self.device)
        return F.linear(z, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class HyperLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        noise_dim: int,
        noise_norm: int = 0,
        prior_std: float = 1.,
        prior_scale: float = 1.,
        posterior_scale: float = 1.,
        hyper_init: str = "xavier_normal",
        prior_init: str = "DB",
        bias_init: str = "sphere",
        device: Optional[Union[str, int, torch.device]] = "cpu",
        **kwargs,
    ) -> None:
        super().__init__()
        hyperlayer_params = dict(
            noise_dim=noise_dim, hidden_dim=in_features, action_dim=out_features, prior_std=prior_std, bias_init=bias_init, device=device
        )
        self.hyper_weight = HyperLayer(**hyperlayer_params, trainable=True, weight_init=hyper_init)
        if prior_std > 0:
            self.prior_weight = HyperLayer(**hyperlayer_params, trainable=False, weight_init=prior_init)

        self.device = device
        self.noise_dim = noise_dim
        self.noise_norm = noise_norm
        self.prior_std = prior_std
        self.posterior_scale = posterior_scale
        self.prior_scale = prior_scale
        self.weight_shape = (in_features, out_features)
        self.bias_shape = (out_features,)

    def base_forward(self, x: torch.Tensor, weight: torch.Tensor):
        weight = weight.reshape((weight.shape[0], -1,) + self.weight_shape) # (batch, noise_num, in_dim, out_dim)
        if weight.shape[0] == x.shape[0]:
            out = torch.einsum('bh, bnha -> bna', x, weight)
        else:
            out = torch.einsum('bh, nha -> bna', x, weight.squeeze(0))
        return out

    def forward(self, x: torch.Tensor, prior_x: torch.Tensor = None, noise: torch.Tensor = None) -> torch.Tensor:
        hyper_noise = self.process_noise(noise)
        weight = self.hyper_weight(hyper_noise)
        out = self.base_forward(x, weight)
        if prior_x is not None and self.prior_std > 0:
            prior_weight = self.prior_weight(hyper_noise)
            prior_out = self.base_forward(prior_x, prior_weight)
            out = out * self.posterior_scale + prior_out * self.prior_scale
        return out

    def process_noise(self, noise: torch.Tensor) -> torch.Tensor:
        noise = noise.to(self.device)
        if self.noise_norm:
            noise /= torch.norm(noise, dim=-1, keepdim=True)
        return noise

    def regularization(self, noise: torch.Tensor) -> torch.Tensor:
        hyper_noise = self.process_noise(noise)
        weight = self.hyper_weight(hyper_noise)
        params = weight
        return params.pow(2).mean()


class HyperNet(BaseNet):
    def __init__(
        self,
        state_shape: Union[int, Sequence[int]],
        action_shape: Union[int, Sequence[int]] = 0,
        hidden_sizes: Sequence[int] = (),
        norm_layer: Optional[ModuleType] = None,
        activation: Optional[ModuleType] = nn.ReLU,
        device: Union[str, int, torch.device] = "cpu",
        softmax: bool = False,
        num_atoms: int = 1,
        base_prior: bool = True,
        use_dueling: bool = False,
        model_type: str = 'mlp',
        normalize: str = '',
        last_layer_params: Dict[str, Any] = None
    ) -> None:
        last_layer = lambda x, y: HyperLinear(x, y, **last_layer_params)
        super().__init__(
            state_shape, action_shape, hidden_sizes, norm_layer, activation, device,
            softmax, num_atoms, use_dueling, base_prior, model_type, normalize,
            last_layer=last_layer,
        )

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        noise: Dict[str, Any] = {},
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: s -> flatten (inside MLP)-> logits."""
        if self.model_type == 'mlp':
            s = s.reshape(-1, self.input_dim)
        hidden_logits, prior_hidden_logits = self.feature(s)
        bsz = hidden_logits.shape[0]
        q = self.Q(hidden_logits, prior_hidden_logits, noise=noise['Q'])
        q = q.view(bsz, -1, self.action_num, self.num_atoms)
        logits = q.squeeze(dim=-1)
        return logits, state
