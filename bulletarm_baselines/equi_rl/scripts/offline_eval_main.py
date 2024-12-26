import os
import sys
import time
import copy
import collections
from tqdm import tqdm
import datetime
import matplotlib
import numpy as np
matplotlib.use('Agg')

# sys.path.append('./')
# sys.path.append('..')
sys.path.append('../../..')
from bulletarm_baselines.equi_rl.utils.parameters import *
from bulletarm_baselines.equi_rl.storage.buffer import QLearningBufferExpert, QLearningBuffer
from bulletarm_baselines.equi_rl.storage.per_buffer import PrioritizedQLearningBuffer, EXPERT, NORMAL
from bulletarm_baselines.equi_rl.storage.aug_buffer import QLearningBufferAug
from bulletarm_baselines.equi_rl.storage.per_aug_buffer import PrioritizedQLearningBufferAug
from bulletarm_baselines.equi_rl.utils.schedules import LinearSchedule
from bulletarm_baselines.equi_rl.utils.env_wrapper import EnvWrapper

from bulletarm_baselines.equi_rl.utils.create_agent import createAgent
import threading

from bulletarm_baselines.equi_rl.utils.torch_utils import ExpertTransition, normalizeTransition, augmentBuffer

from bulletarm_baselines.logger.baseline_logger import BaselineLogger

def set_seed(s):
    print("Setting Seed: {}".format(s))
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def train_step(agent, replay_buffer,n_training_steps):
    batch = replay_buffer.sample(batch_size)
    loss, td_error = agent.update(batch)

    if n_training_steps % target_update_freq == 0:
        agent.updateTarget()



def evaluate(envs, agent):
    states, obs = envs.reset()
    evaled = 0

    temp_reward = [[] for _ in range(num_eval_processes)]
    eval_rewards = []
    
    if not no_bar:
        eval_bar = tqdm(total=num_eval_episodes)
    
    while evaled < num_eval_episodes:
        actions_star_idx, actions_star = agent.getGreedyActions(states, obs)
        states_, obs_, rewards, dones = envs.step(actions_star, auto_reset=True)
        rewards = rewards.numpy()
        dones = dones.numpy()
        states = copy.copy(states_)
        obs = copy.copy(obs_)
        for i, r in enumerate(rewards.reshape(-1)):
            temp_reward[i].append(r)
        evaled += int(np.sum(dones))
        for i, d in enumerate(dones.astype(bool)):
            if d:
                R = 0
                for r in reversed(temp_reward[i]):
                    R = r + gamma * R
                eval_rewards.append(R)
                temp_reward[i] = []
        
        
        if not no_bar:
            eval_bar.update(evaled - eval_bar.n)


    eval_rewards_numpy = np.array(eval_rewards)
    mean_reward = np.mean(eval_rewards_numpy)
    if not no_bar:
        eval_bar.close()
    

    print("Evaluated reward: {}".format(mean_reward))


def train():
    eval_thread = None
    start_time = time.time()
    if seed is not None:
        set_seed(seed)
    # setup env
    print('creating envs')
    # num_eval_processes = 1
    envs = EnvWrapper(num_processes, env, env_config, planner_config)
    eval_envs = EnvWrapper(num_eval_processes, env, env_config, planner_config)

    # setup agent
    agent = createAgent()
    eval_agent = createAgent(test=True)
    # .train() is required for equivariant network
    agent.train()
    eval_agent.train()

    
    load_model_dir = "/home/arsh/Desktop/Equivariant-Offline-RL/bulletarm_baselines/equi_rl/scripts/models/close_loop_block_stacking_expert_buffer_10/bcfd/model_26000_0.47285949863405585"


    agent.loadModel(load_model_dir)
    print("Model loaded from: {}".format(load_model_dir))

    hyper_parameters['model_shape'] = agent.getModelStr()
    replay_buffer = QLearningBuffer(buffer_size)
    



    if not no_bar:
        pbar = tqdm(total=max_train_step)
        pbar.set_description('Episodes:0; Reward:0.0;Time: 0.0')
    timer_start = time.time()

    states, obs = envs.reset()
    num_training_steps = 0
    episode_rewards = []
    episode_reward = [0 for _ in range(num_processes)]
    current_eval_reward = 0



    if eval_thread is not None:
        eval_thread.join()
    eval_agent.copyNetworksFrom(agent)
    eval_thread = threading.Thread(target=evaluate, args=(eval_envs, eval_agent))
    eval_thread.start()
    eval_thread.join()


    envs.close()
    eval_envs.close()
    print('training finished')
    if not no_bar:
        pbar.close()

if __name__ == '__main__':
    train()