import argparse
import os
import time
import json
import pprint
import numpy as np
import torch

from hyperfqi.data import Collector, VectorReplayBuffer
from hyperfqi.env import DummyVectorEnv
from hyperfqi.env.utils import make_atari_env, make_atari_env_watch
from hyperfqi.trainer import offpolicy_trainer
from hyperfqi.utils import TensorboardLogger
from hyperfqi.network.hyper_network import HyperNet
from hyperfqi.policy.hyper_policy import  HyperPolicy


def get_args():
    parser = argparse.ArgumentParser()
    # environment config
    parser.add_argument('--task', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--frames-stack', type=int, default=4)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=10)
    # training config
    parser.add_argument('--max-target', type=float, default=0.8)
    parser.add_argument('--target-noise-per-sample', type=int, default=5)
    parser.add_argument('--update-noise-per-sample', type=int, default=20)
    parser.add_argument('--target-update-freq', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr-eps', type=float, default=0.00015)
    parser.add_argument('--weight-decay', type=float, default=0.)
    parser.add_argument('--hyper-weight-decay', type=float, default=0.)
    parser.add_argument('--n-step', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--clip-grad-norm', type=float, default=10.)
    # algorithm config
    parser.add_argument('--alg-type', type=str, default="HyperFQI")
    parser.add_argument('--noise-norm', type=int, default=1, choices=[0, 1])
    parser.add_argument('--noise-std', type=float, default=1.)
    parser.add_argument('--noise-dim', type=int, default=4, help="Greater than 0 means using HyperModel")
    parser.add_argument('--prior-std', type=float, default=0.1, help="Greater than 0 means using priormodel")
    parser.add_argument('--prior-scale', type=float, default=0.1)
    parser.add_argument('--posterior-scale', type=float, default=0.1)
    parser.add_argument('--target-noise-coef', type=float, default=0.01)
    parser.add_argument('--target-noise-type', type=str, default="sp", choices=["sp", "gs", "bi"])
    parser.add_argument('--hyper-reg-coef', type=float, default=0.01)
    parser.add_argument('--grad-coef', type=float, default=0.01)
    parser.add_argument('--one-hot-noise', type=int, default=0, choices=[0, 1])
    # network config
    parser.add_argument('--hidden-layer', type=int, default=1)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--hyper-init', type=str, default="xavier_normal", choices=["sDB", "gDB", "xavier_normal"])
    parser.add_argument('--prior-init', type=str, default="sDB", choices=["sDB", "gDB", "xavier_normal"])
    parser.add_argument('--bias-init', type=str, default="xavier-uniform", choices=["sphere", "xavier_normal"])
    # epoch config
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--step-per-epoch', type=int, default=10000)
    parser.add_argument('--step-per-collect', type=int, default=1)
    parser.add_argument('--update-per-step', type=float, default=1)
    parser.add_argument('--episode-per-test', type=int, default=10)
    # buffer confing
    parser.add_argument('--buffer-size', type=int, default=int(1e5))
    parser.add_argument('--min-buffer-size', type=int, default=2000)
    # action selection confing
    parser.add_argument('--random-start', type=int, default=1, choices=[0, 1], help="1: use random policy to collect minibuffersize data")
    parser.add_argument('--action-sample-num', type=int, default=1)
    parser.add_argument('--action-select-scheme', type=str, default="Greedy", choices=['Greedy', 'MAX'])
    parser.add_argument('--quantile-max', type=float, default=0.8)
    # other confing
    parser.add_argument('--logdir', type=str, default='~/results/hyperfqi/atari')
    parser.add_argument('--evaluation', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--version', type=str, default='sprconfig feature network: xavier_normal')
    args = parser.parse_known_args()[0]
    return args


def main(args=get_args()):
    # environment
    train_envs = DummyVectorEnv([lambda: make_atari_env(args) for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([lambda: make_atari_env_watch(args) for _ in range(args.test_num)])
    args.state_shape = train_envs.observation_space[0].shape or train_envs.observation_space[0].n
    args.action_shape = train_envs.action_space[0].shape or train_envs.action_space[0].n

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    if args.device == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(args.seed)

    # model
    last_layer_params = {
        'device': args.device,
        'noise_dim': args.noise_dim,
        'noise_norm': args.noise_norm,
        'prior_std': args.prior_std,
        'prior_scale': args.prior_scale,
        'posterior_scale': args.posterior_scale,
        'hyper_init': args.hyper_init,
        'prior_init': args.prior_init,
        'bias_init': args.bias_init,
    }
    args.hidden_sizes = [args.hidden_size] * args.hidden_layer
    model_params = {
        "state_shape": args.state_shape,
        "action_shape": args.action_shape,
        "hidden_sizes": args.hidden_sizes,
        "device": args.device,
        "model_type": "conv",
        "normalize": "MinMax",
        "last_layer_params": last_layer_params
    }
    model = HyperNet(**model_params).to(args.device)

    # init base model
    for name, param in model.named_parameters():
        if name.startswith('feature'):
            if 'bias' in name:
                torch.nn.init.zeros_(param)
            elif 'weight' in name:
                torch.nn.init.xavier_normal_(param, gain=1.0)

    param_dict = {"Non-trainable": [], "Trainable": []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            param_dict["Non-trainable"].append(name)
        else:
            param_dict["Trainable"].append(name)
    pprint.pprint(param_dict)
    print(f"Network structure:\n{str(model)}")
    print(f"Network parameters: {sum(param.numel() for param in model.parameters())}")

    # optimizer
    trainable_params = [
            {
                'params': (p for name, p in model.named_parameters() if 'prior' not in name and 'hyper' in name),
                'weight_decay': args.hyper_reg_coef,
            },
            {
                'params': (p for name, p in model.named_parameters() if 'prior' not in name and 'based' in name),
                'weight_decay': args.weight_decay,
            },
        ]
    optim = torch.optim.Adam(trainable_params, lr=args.lr, eps=args.lr_eps)

    # policy
    policy_params = {
        "model": model,
        "optim": optim,
        "discount_factor": args.gamma,
        "estimation_step": args.n_step,
        "target_update_freq": args.target_update_freq,
        "max_target": args.max_target,
        "update_noise_per_sample": args.update_noise_per_sample,
        "target_noise_per_sample": args.target_noise_per_sample,
        "action_sample_num": args.action_sample_num,
        "action_select_scheme": args.action_select_scheme,
        "quantile_max": args.quantile_max,
        "noise_std": args.noise_std,
        "noise_dim": args.noise_dim,
        "grad_coef": args.grad_coef,
        "hyper_reg_coef": args.hyper_reg_coef,
        "target_noise_coef": args.target_noise_coef,
        "target_noise_type": args.target_noise_type,
        "clip_grad_norm": args.clip_grad_norm,
        "one_hot_noise": args.one_hot_noise,
        "seed": args.seed,
    }
    policy = HyperPolicy(**policy_params).to(args.device)

    # buffer
    buf = VectorReplayBuffer(
        args.buffer_size, buffer_num=len(train_envs),
        ignore_obs_next=True, save_only_last_obs=True, stack_num=args.frames_stack
    )

    # collector
    args.target_noise_dim = args.noise_dim
    train_collector = Collector(
        policy, train_envs, buf, exploration_noise=False,
        target_noise_dim=args.target_noise_dim, target_noise_type=args.target_noise_type,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=False)

    # log
    args.logfile = f"{args.alg_type}_{args.task}_{args.seed}_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
    log_path = os.path.join(args.logdir, args.task, args.logfile)
    log_path = os.path.expanduser(log_path)
    os.makedirs(log_path, exist_ok=True)
    logger = TensorboardLogger(log_path)
    with open(os.path.join(log_path, "config.json"), "wt") as f:
        kvs = vars(args)
        f.write(json.dumps(kvs, indent=4) + '\n')
        f.flush()
        f.close()

    def save_fn(policy, env_step=None):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    # trainer
    args.step_per_collect *= args.training_num
    args.update_per_step *= args.training_num
    args.update_per_step /= args.step_per_collect
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.episode_per_test,
        batch_size=args.batch_size,
        learning_start=args.min_buffer_size,
        random_start=args.random_start,
        update_per_step=args.update_per_step,
        save_fn=save_fn,
        logger=logger,
        verbose=True,
    )
    # assert stop_fn(result['best_reward'])
    # pprint.pprint(result)
    if args.evaluation > 0:
        eval_envs = DummyVectorEnv(
            [lambda: make_atari_env_watch(args) for _ in range(args.evaluation // 10)]
        )
        # reset seed
        policy.reset_test_noise(args.seed)
        eval_envs.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed(args.seed)
        test(log_path, eval_envs, policy, args.evaluation)


def test(log_path, env, policy: torch.nn.Module, n_episode: int):
    from hyperfqi.utils.logger.tensorboard import CSVOutputFormat

    policy.load_state_dict(torch.load(os.path.join(log_path, 'policy.pth')))
    collector = Collector(policy, env)
    best_results = collector.collect(n_episode=n_episode)

    results = {
        "best/rew": best_results["rew"],
        "best/rew_std": best_results["rew_std"],
    }
    with open(os.path.join(log_path, "results.json"), "wt") as f:
        f.write(json.dumps(results, indent=4) + '\n')
        f.flush()
        f.close()

    logger = CSVOutputFormat(os.path.join(log_path, 'results.csv'))
    for best_rew in best_results['rews']:
        logger.writekvs({"best/rew": best_rew})


if __name__ == '__main__':
    args = get_args()
    main(args)
