from typing import Any, Dict, Optional, Union

import torch
import torch.nn.functional as F
import numpy as np

from hyperfqi.policy import DQNPolicy
from hyperfqi.data import Batch, ReplayBuffer, to_torch_as, to_numpy
from hyperfqi.policy.hyper_utils import SampleNoise, ActionSelection, BootstrappedNoise, rd_argmax


class HyperPolicy(DQNPolicy):
    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        discount_factor: float = 0.99,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        is_double: bool = True,
        max_target: float = 0.0,
        update_noise_per_sample: int = 1,
        target_noise_per_sample: int = 1,
        action_sample_num: int = 1,
        action_select_scheme: str = "Greedy",
        quantile_max: float = 1.0,
        noise_std: float = 1.,
        noise_dim: int = 4,
        grad_coef: float = 0,
        hyper_reg_coef: float = 0.01,
        target_noise_coef: float = 0.01,
        target_noise_type: str = 'sp',
        clip_grad_norm: float = 10.,
        one_hot_noise: bool = False,
        seed: int = 2022,
        **kwargs: Any
    ) -> None:
        super().__init__(
            model, optim, discount_factor, estimation_step, target_update_freq, reward_normalization, is_double, **kwargs
        )
        if action_select_scheme == "Greedy":
            assert action_sample_num == 1
        self.max_target = max_target
        self.target_noise_per_sample = target_noise_per_sample
        self.update_noise_per_sample = update_noise_per_sample
        self.action_sample_num = action_sample_num
        self.action_select_scheme = action_select_scheme
        self.grad_coef = grad_coef
        if grad_coef > 0:
            self.param_list = [param for param in self.model.parameters() if param.requires_grad]
        self.hyper_reg_coef = hyper_reg_coef
        self.target_noise_coef = target_noise_coef
        self.target_noise_type = target_noise_type
        self.clip_grad_norm = clip_grad_norm
        self.noise_test = None
        self.noise_train = None
        self.noise_update = None
        self.noise_target = None
        self.sample_noise = SampleNoise(noise_dim, noise_std, one_hot_noise, seed=seed)
        self.sample_target_noise = self.sample_noise.sample_target_noise
        self.sample_update_noise = self.sample_noise.sample_update_noise
        self.sample_train_noise = self.sample_noise.sample_train_noise
        self.sample_test_noise = self.sample_noise.sample_test_noise
        self.get_actions = ActionSelection(action_select_scheme, quantile_max).get_actions
        self.compute_noise = BootstrappedNoise(self.target_noise_type, self.target_noise_coef).compute_noise

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]  # batch.obs_next: s_{t+n}
        # compute target q
        if self._is_double:
            with torch.no_grad():
                main_q = self(batch, model="model", input="obs_next", noise=self.noise_target).logits # (None, N, action_num)
                target_q = self(batch, model="model_old", input="obs_next", noise=self.noise_target).logits # (None, N, action_num)
            if self.target_noise_per_sample > 1:
                main_q = torch.quantile(main_q, self.max_target, dim=1, keepdim=True)
                target_q = torch.quantile(target_q, self.max_target, dim=1, keepdim=True)
            main_q, target_q = main_q.squeeze(1), target_q.squeeze(1)
            main_q = to_numpy(main_q)
            act = [rd_argmax(main_q[i]) for i in range(main_q.shape[0])]
            target_q = target_q[np.arange(len(act)), act].unsqueeze(1)
        else:
            raise NotImplementedError
        return target_q

    def forward(
        self, batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        noise: Dict[str, Any] = None,
        **kwargs: Any
    ) -> Batch:
        model = getattr(self, model)
        obs = batch[input]
        obs_ = obs.obs if hasattr(obs, "obs") else obs
        if not self.updating:
            if self.training:
                if self.noise_train is None:
                    self.noise_train = self.sample_train_noise((obs_.shape[0], self.action_sample_num))
                else:
                    change_tag = np.random.binomial(size=obs_.shape[0], n=1, p=self._gamma)
                    change_id = np.where(change_tag == 0)[0]
                    done_id = np.where(batch.done)[0]
                    noise_id = np.unique(np.hstack([change_id, done_id]))
                    new_noise = self.sample_train_noise((len(noise_id), self.action_sample_num))['Q']
                    self.noise_train['Q'][noise_id] = new_noise
                noise = self.noise_train
            else:
                if self.noise_test is None or len(batch.done.shape) == 0:
                    self.noise_test = self.sample_test_noise((obs_.shape[0], self.action_sample_num))
                elif np.any(batch.done):
                    done_id = np.where(batch.done)[0]
                    new_noise = self.sample_test_noise((len(done_id), self.action_sample_num))['Q']
                    self.noise_test['Q'][done_id] = new_noise
                noise = {key: val[batch.env_ids] for key, val in self.noise_test.items()}
            logits, h = model(obs_, state=state, noise=noise, info=batch.info) # (None)
            q = self.compute_q_value(logits, getattr(obs, "mask", None))  # (None, noise_num)
            act = self.get_actions(q)
        else:
            logits, h = model(obs_, state=state, noise=noise, info=batch.info) # (None)
            q = self.compute_q_value(logits, getattr(obs, "mask", None))  # (None, noise_num)
            act = to_numpy(q.max(dim=-1)[1])
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[-1]
        return Batch(logits=logits, act=act, state=h)

    def update(self, sample_size: int, buffer: Optional[ReplayBuffer], **kwargs: Any) -> Dict[str, Any]:
        # update target network
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()
        # sample noise for main network
        self.noise_update = self.sample_update_noise((sample_size, self.update_noise_per_sample))
        # sample noise for target network
        self.noise_target = self.sample_target_noise((sample_size, self.target_noise_per_sample))
        return super().update(sample_size, buffer, **kwargs)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        results = {}
        if self.target_noise_coef:
            loss_noise = self.compute_noise(self.noise_update['Q'], batch.target_noise)
        q = self(batch, noise=self.noise_update).logits # (None, action_num)
        a_one_hot = F.one_hot(torch.from_numpy(batch.act), self.max_action_num).to(torch.float32) # (None, num_actions)
        q = torch.einsum('bka,ba->bk', q, a_one_hot.to(q.device)) # (None, noise_per_sample)
        r = to_torch_as(batch.returns, q) # (None, noise_per_sampe)
        if self.target_noise_coef and self.target_noise_type in [ "sp", "gs"]:
            r = r + to_torch_as(loss_noise, r)
        td = (r - q).pow(2) # (None, noise_per_sample)
        if self.target_noise_coef and self.target_noise_type in ["bi"]:
            td *= to_torch_as(loss_noise, td)
        td = td.mean(-1) # (None,)
        loss = td.mean()

        if self.grad_coef > 0:
            grads = torch.autograd.grad(
                outputs=q.mean(-1),
                inputs=self.param_list,
                grad_outputs=torch.ones(q.mean(-1).size()).to(q.device),
                create_graph=True,
                retain_graph=True,
            )
            grad_loss = 0
            for grad in grads:
                grad_loss += torch.norm(grad/len(batch)).pow(2)
            loss = loss + self.grad_coef * grad_loss
            results.update({"grad_loss": grad_loss.item()})

        if self.hyper_reg_coef:
            reg_loss = self.model.Q.regularization(self.noise_update['Q'])
            loss += reg_loss * (self.hyper_reg_coef / kwargs['sample_num'])
            for param_group in self.optim.param_groups:
                param_group['weight_decay'] /= kwargs['sample_num']

        self.optim.zero_grad()
        loss.backward()
        if self.clip_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            results.update({"grad_norm": grad_norm.item()})
        self.optim.step()

        self._iter += 1
        results.update({"loss": loss.item()})
        return results

    def reset_test_noise(self, seed: int):
        self.noise_test = None
        self.sample_noise.reset_seed(seed, "test")
