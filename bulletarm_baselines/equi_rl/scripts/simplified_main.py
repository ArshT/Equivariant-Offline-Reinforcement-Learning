'''
Main script (Simplified) for training the agent using any Online-RL algorithm supported by BulletArm Baselines.
'''


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

import wandb

def set_seed(s):
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



def evaluate(envs, agent,result_list):
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

    result_list[0] = mean_reward


def train():
    eval_thread = None
    start_time = time.time()
    if seed is not None:
        set_seed(seed)
    # setup env
    print('creating envs')
    envs = EnvWrapper(num_processes, env, env_config, planner_config)
    eval_envs = EnvWrapper(num_eval_processes, env, env_config, planner_config)

    # setup agent
    agent = createAgent()
    eval_agent = createAgent(test=True)
    # .train() is required for equivariant network
    agent.train()
    eval_agent.train()



    wandb.login()

    run = wandb.init(
        # Set the project where this run will be logged
        project="Equi_Offline_RL",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "cql_weight": cql_weight,
            "cql_temperature": cql_temperature,
            "tau": tau,
            "batch_size": batch_size,
            "algorithm": algorithm,
            "env": env,
            "seed": seed,           
            "Task": "Online Finetuning",
        },
    )




    if planner_episode > 0:
        print("Using Half Expert Buffer")
        replay_buffer = QLearningBufferExpert(buffer_size)
    else:
        replay_buffer = QLearningBuffer(buffer_size)
    
    
    if planner_episode > 0:
        planner_envs = envs
        result = [None]
        planner_num_process = num_processes
        j = 0
        states, obs = planner_envs.reset()
        s = 0
        if not no_bar:
            planner_bar = tqdm(total=planner_episode)
        while j < planner_episode:
            plan_actions = planner_envs.getNextAction()
            planner_actions_star_idx, planner_actions_star = agent.getActionFromPlan(plan_actions)
            states_, obs_, rewards, dones = planner_envs.step(planner_actions_star, auto_reset=True)
            for i in range(planner_num_process):
                transition = ExpertTransition(states[i].numpy(), obs[i].numpy(), planner_actions_star_idx[i].numpy(),
                                            rewards[i].numpy(), states_[i].numpy(), obs_[i].numpy(), dones[i].numpy(),
                                            np.array(100), np.array(1))
                if obs_type == 'pixel':
                    transition = normalizeTransition(transition)
                replay_buffer.add(transition)
            states = copy.copy(states_)
            obs = copy.copy(obs_)

            j += dones.sum().item()
            s += rewards.sum().item()

            if not no_bar:
                planner_bar.set_description('{:.3f}/{}, AVG: {:.3f}'.format(s, j, float(s)/j if j != 0 else 0))
                planner_bar.update(dones.sum().item())


        if not no_bar:
            pbar = tqdm(total=max_train_step)
            pbar.set_description('Episodes:0; Reward:0.0;Time: 0.0')
        timer_start = time.time()

        states, obs = envs.reset()
        num_training_steps = 0
        episode_rewards = []
        episode_reward = [0 for _ in range(num_processes)]
        current_eval_reward = 0


        while num_training_steps < max_train_step:
            is_expert = 0
            actions_star_idx, actions_star = agent.getEGreedyActions(states, obs,eps=0)

            envs.stepAsync(actions_star, auto_reset=False)
            num_training_steps += num_processes

            if len(replay_buffer) >= training_offset:
                for training_iter in range(training_iters):
                    train_step(agent, replay_buffer,num_training_steps)

            states_, obs_, rewards, dones = envs.stepWait()

            done_idxes = torch.nonzero(dones).squeeze(1)
            if done_idxes.shape[0] != 0:
                reset_states_, reset_obs_ = envs.reset_envs(done_idxes)
                for j, idx in enumerate(done_idxes):
                    states_[idx] = reset_states_[j]
                    obs_[idx] = reset_obs_[j]

                    episode_rewards.append(episode_reward[idx])
                    episode_reward[idx] = 0

                    

            
            for i in range(num_processes):
                transition = ExpertTransition(states[i].numpy(), obs[i].numpy(), actions_star_idx[i].numpy(),
                                                rewards[i].numpy(), states_[i].numpy(), obs_[i].numpy(), dones[i].numpy(),
                                                np.array(100), np.array(is_expert))
                if obs_type == 'pixel':
                    transition = normalizeTransition(transition)
                replay_buffer.add(transition)
            
            
            
            for i in range(num_processes):
                episode_reward[i] += rewards[i]

            
            states = copy.copy(states_)
            obs = copy.copy(obs_)

            if (time.time() - start_time)/3600 > time_limit:
                break

            if not no_bar:
                timer_final = time.time()
                # description = 'Action Step:{}; Reward:{:.03f}; Eval Reward:{:.03f}; Explore:{:.02f}; Loss:{:.03f}; Time:{:.03f}'.format(
                #     logger.num_steps, logger.getAvg(logger.training_eps_rewards, 100), np.mean(logger.eval_eps_rewards[-2]) if len(logger.eval_eps_rewards) > 1 and len(logger.eval_eps_rewards[-2]) > 0 else 0, eps, float(logger.getCurrentLoss()),
                #     timer_final - timer_start)

                description = 'Action Step:{}; Reward:{:.03f}; Time:{:.03f}'.format(
                    num_training_steps, np.mean(episode_rewards[-100:]) if len(episode_rewards) > 0 else 0, current_eval_reward,
                    timer_final - timer_start)
                
                pbar.set_description(description)
                timer_start = timer_final
                pbar.update(num_training_steps-pbar.n)


            if num_training_steps > 0 and eval_freq > 0 and num_training_steps % eval_freq == 0:
                if eval_thread is not None:
                    eval_thread.join()
                eval_agent.copyNetworksFrom(agent)
                eval_thread = threading.Thread(target=evaluate, args=(eval_envs, eval_agent,result))
                eval_thread.start()
                print("Received Evaluated reward: {}".format(result[0]))


            if result[0] != None:
                wandb.log({"Evaluated Reward": result[0]}, step=num_training_steps)


            


    envs.close()
    eval_envs.close()
    print('training finished')
    if not no_bar:
        pbar.close()

if __name__ == '__main__':
    train()
