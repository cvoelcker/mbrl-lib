# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# @author: Claas Voelcker

import os
from typing import Dict, Optional, Tuple, cast

import gym
import hydra.utils
from mbrl.env.offline_data import load_dataset_and_env
import numpy as np
import omegaconf
from omegaconf import read_write
import torch

import mbrl.constants
import mbrl.models
import mbrl.planning
import mbrl.third_party.pytorch_sac as pytorch_sac
import mbrl.types
import mbrl.util
import mbrl.util.common
import mbrl.util.math
from mbrl.planning.sac_wrapper import SACAgent

MBPO_LOG_FORMAT = mbrl.constants.EVAL_LOG_FORMAT + [
    ("epoch", "E", "int"),
    ("rollout_length", "RL", "int"),
]


def create_dataloader_from_dict(cfg: omegaconf.DictConfig, env: gym.Env, dataset: Dict) -> mbrl.util.ReplayBuffer:

    dataset_length = len(dataset["observations"])
    assert cfg.overrides.num_steps >= dataset_length, \
        f"Buffer must be large enough for pretraining dataset, trying to fit {dataset_length} into {cfg.overrides.num_steps} steps."

    rng = np.random.default_rng(seed=cfg.seed)
    dtype = np.float32
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    replay_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )
    
    observations = dataset["observations"]
    actions = dataset["actions"]
    rewards = dataset["rewards"]
    next_observations = dataset["next_observations"]
    dones = dataset["terminals"]
    if "timeouts" in dataset.keys():
        dones = np.logical_or(dataset["terminals"], dataset["timeouts"])

    for (obs, act, rew, obs_next, done) in zip(observations, actions, rewards, next_observations, dones):
        replay_buffer.add(obs, act, obs_next, rew, done)

    return replay_buffer


def train(
    env: gym.Env,
    cfg: omegaconf.DictConfig,
) -> None:
    # ------------------- Initialization -------------------
    assert cfg.model_pretraining.train_dataset == cfg.overrides.env, "Dataset for pretraining must come from the training env."

    debug_mode = cfg.get("debug_mode", False)

    train_dataset, _ = load_dataset_and_env(cfg.model_pretraining.train_dataset)
    test_dataset, _ = load_dataset_and_env(cfg.model_pretraining.test_dataset)

    train_dataset = create_dataloader_from_dict(cfg, env, train_dataset)
    test_dataset = create_dataloader_from_dict(cfg, env, test_dataset)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    # mbrl.planning.complete_agent_cfg(env, cfg.algorithm.agent)
    # agent = hydra.utils.instantiate(cfg.algorithm.agent)

    work_dir = os.getcwd()

    logger = mbrl.util.Logger(work_dir, enable_back_compatible=True)
    logger.register_group(
        mbrl.constants.RESULTS_LOG_NAME,
        MBPO_LOG_FORMAT,
        color="green",
        dump_frequency=1,
    )

    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    model_trainer = mbrl.models.ModelTrainer(
        dynamics_model,
        optim_lr=cfg.overrides.model_lr,
        weight_decay=cfg.overrides.model_wd,
        logger=logger,
    )
    mbrl.util.common.train_model_and_save_model_and_data(
        dynamics_model,
        model_trainer,
        cfg.overrides,
        train_dataset,
        work_dir=work_dir,
    )

    return dynamics_model, model_trainer, train_dataset

    # ---------------------------------------------------------
    # -------------- Evaluate on test dataset -----------------

    pass
    #TODO: implement more mature testing logic here and maybe some nice viz

    # ---------------------------------------------------------
    # -------------- Create initial overrides. dataset --------------

    pass
    #TODO: implement robust saving so we can use this model for more experiments
    # down the line