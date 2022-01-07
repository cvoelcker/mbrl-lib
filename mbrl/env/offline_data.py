from typing import Dict, Tuple
import gym
import d4rl # Import required to register environments


def load_dataset_and_env(env_name: str) -> Tuple[Dict, gym.Env]:
    # Create the environment
    if "d4rl___" in env_name:
        env = gym.make(env_name.split("___")[1])
    else:
        env = gym.make(env_name)
    env.reset()
    env.step(env.action_space.sample())
    dataset = d4rl.qlearning_dataset(env)
    return dataset, env
