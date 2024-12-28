'''
Python script to collect Datasets using EquiSAC / EquiSACfD or an expert-planner
Used to collect datasets with a specific number of episodes for optimal datasets. 
Used to collect datasets with a specific number of episodes or a specific number of transitions for sub-optimal datasets. 
Works for both pixel and vector observations.
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
import random
matplotlib.use('Agg')


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

from bulletarm_baselines.equi_rl.utils.torch_utils import ExpertTransition, normalizeTransition, augmentBuffer, ExpertTransitionGoal
from bulletarm_baselines.logger.baseline_logger import BaselineLogger



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



def evaluate(envs, evaluation_agent,result_list,stop_train_list):
    states, obs = envs.reset()
    evaled = 0
    temp_reward = [[] for _ in range(num_eval_processes)]
    eval_rewards = []
    
    if not no_bar:
        eval_bar = tqdm(total=num_eval_episodes)
    
    while evaled < num_eval_episodes:
        actions_star_idx, actions_star = evaluation_agent.getGreedyActions(states, obs)
        
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

    if result_list[0] > data_reward_limit:
        evaluation_agent.copyNetworksFrom(evaluation_agent)
        stop_train_list[0] = True
        print("Training Stopped")
        print('##########################################')




def train():
    eval_thread = None
    result = [0]
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


    os.makedirs("datasets/", exist_ok=True)

    

    hyper_parameters['model_shape'] = agent.getModelStr()
    replay_buffer = QLearningBuffer(buffer_size)
    replay_buffer_mix = QLearningBuffer(offline_buffer_size)
    replay_buffer_sub_optimal = QLearningBuffer(offline_buffer_size)
    
    if data_collection_type == 'optimal':
        planner_episode = data_expert_demos
    
    if planner_episode > 0:
        planner_envs = envs
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
    

    if data_collection_type == 'optimal':

        if obs_type != 'pixel':
            print("Saving: {}".format("datasets/" + env + '_expert_buffer_'+str(data_expert_demos) + "_" + "vector" + "_" + str(seed)))
            replay_buffer.saveBuffer("datasets/" + env + '_expert_buffer_'+str(data_expert_demos) + "_" + "vector" + "_" + str(seed))
            print("Buffer Loaded, Size: {}".format(len(replay_buffer)))
        else:
            print("Saving: {}".format("datasets/" + env + '_expert_buffer_'+str(data_expert_demos) + "_" + str(seed)))    
            replay_buffer.saveBuffer("datasets/" + env + '_expert_buffer_'+str(data_expert_demos) + "_" + str(seed))
            print("Buffer Loaded, Size: {}".format(len(replay_buffer)))

        print("#######################################")
        print("Finishing Data Collection: Optimal")
        print("#######################################")
        return



    if not no_bar:
        pbar = tqdm(total=max_train_step)
        pbar.set_description('Episodes:0; Reward:0.0;Time: 0.0')
    timer_start = time.time()

    states, obs = envs.reset()
    num_training_steps = 0
    episode_rewards = []
    episode_reward = [0 for _ in range(num_processes)]
    current_eval_reward = 0

    buffer_episode_rewards = []
    buffer_episode_number = 0

    stop_training_list = [False]

    while num_training_steps < max_train_step:
        is_expert = 0
        if not stop_training_list[0]:
            actions_star_idx, actions_star = agent.getEGreedyActions(states, obs,eps=0)
        else:
            actions_star_idx, actions_star = agent.getGreedyActions(states, obs)


        envs.stepAsync(actions_star, auto_reset=False)
        num_training_steps += num_processes

        
        if data_collection_type == 'sub_optimal':
            if not data_collection_transitions:
                if buffer_episode_number >= data_demos:
                    print("Sub Optimal Replay Buffer Size: {}".format(len(replay_buffer_sub_optimal)))
                    avg_buffer_episode_reward = np.mean(buffer_episode_rewards)

                    if obs_type != 'pixel':
                        replay_buffer_sub_optimal.saveBuffer("datasets/" + env + '_sub_optimal_buffer_'+str(data_reward_limit)+ "_" + "vector" + "_"+ str(avg_buffer_episode_reward)+ "_" + str(buffer_episode_number) + "_"  + str(seed))
                    else:
                        replay_buffer_sub_optimal.saveBuffer("datasets/" + env + '_sub_optimal_buffer_'+str(data_reward_limit)+ "_"+ str(avg_buffer_episode_reward)+ "_" + str(buffer_episode_number) + "_" + str(seed))
                
                    print("#######################################")
                    print("Finishing Data Collection: Sub Optimal")
                    print("#######################################")

                    break
            else:
                if len(replay_buffer_sub_optimal) >= offline_buffer_size:
                    print("Sub Optimal Replay Buffer Size: {}".format(len(replay_buffer_sub_optimal)))
                    avg_buffer_episode_reward = np.mean(buffer_episode_rewards)

                    if obs_type != 'pixel':
                        replay_buffer_sub_optimal.saveBuffer("datasets/" + env + '_sub_optimal_buffer_'+str(data_reward_limit) + "_" + "vector" + "_"+ str(avg_buffer_episode_reward)+ "_" + str(buffer_episode_number)  + "_" + str(offline_buffer_size) + "_" + str(seed))
                    else:
                        replay_buffer_sub_optimal.saveBuffer("datasets/" + env + '_sub_optimal_buffer_'+str(data_reward_limit)+ "_"+ str(avg_buffer_episode_reward)+ "_" + str(buffer_episode_number) + "_" + str(offline_buffer_size) + "_" + str(seed))

                    print("#######################################")
                    print("Finishing Data Collection: Sub-Optimal")
                    print("#######################################")

                    break



        if len(replay_buffer) >= training_offset:
            if not stop_training_list[0]:
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

                if stop_training_list[0]:

                    if idx == 0:    
                        buffer_episode_rewards.append(episode_reward[0])
                        buffer_episode_number += 1

                episode_reward[idx] = 0

                

        
        for i in range(num_processes):
            transition = ExpertTransition(states[i].numpy(), obs[i].numpy(), actions_star_idx[i].numpy(),
                                            rewards[i].numpy(), states_[i].numpy(), obs_[i].numpy(), dones[i].numpy(),
                                            np.array(100), np.array(is_expert))
            if obs_type == 'pixel':
                transition = normalizeTransition(transition)
            replay_buffer.add(transition)

            if stop_training_list[0]:

                if i == 0:
                    if data_collection_type == 'mix':
                        replay_buffer_mix.add(transition)
                    elif data_collection_type == 'sub_optimal':
                        replay_buffer_sub_optimal.add(transition)
        
        
        for i in range(num_processes):
            episode_reward[i] += rewards[i]

        
        states = copy.copy(states_)
        obs = copy.copy(obs_)

        if (time.time() - start_time)/3600 > time_limit:
            break

        if not no_bar:
            timer_final = time.time()

            description = 'Action Step:{}; Reward:{:.03f}; Time:{:.03f}'.format(
                num_training_steps, np.mean(episode_rewards[-100:]) if len(episode_rewards) > 0 else 0, current_eval_reward,
                timer_final - timer_start)
            
            pbar.set_description(description)
            timer_start = timer_final
            pbar.update(num_training_steps-pbar.n)

        
        if num_training_steps > 0 and eval_freq > 0 and num_training_steps % eval_freq == 0 and not stop_training_list[0]:
            if eval_thread is not None:
                eval_thread.join()
            
            # eval_agent.copyNetworksFrom(agent)
            eval_thread = threading.Thread(target=evaluate, args=(eval_envs, agent,result,stop_training_list))
            eval_thread.start()
            print("Received Evaluated reward: {}".format(result[0]))


            


    envs.close()
    eval_envs.close()
    print('training finished')
    if not no_bar:
        pbar.close()

if __name__ == '__main__':
    train()