import os
import sys
import time
import copy
import collections
from tqdm import tqdm
import datetime
import matplotlib
import numpy as np
import wandb

matplotlib.use('Agg')


sys.path.append('../../..')
from bulletarm_baselines.equi_rl.utils.parameters import *
from bulletarm_baselines.equi_rl.storage.buffer import QLearningBufferExpert, QLearningBuffer
from bulletarm_baselines.equi_rl.storage.per_buffer import PrioritizedQLearningBuffer, EXPERT, NORMAL
from bulletarm_baselines.equi_rl.storage.aug_buffer import QLearningBufferAug
from bulletarm_baselines.equi_rl.storage.per_aug_buffer import PrioritizedQLearningBufferAug
from bulletarm_baselines.equi_rl.utils.schedules import LinearSchedule
from bulletarm_baselines.equi_rl.utils.env_wrapper import EnvWrapper
from queue import PriorityQueue

from bulletarm_baselines.equi_rl.utils.create_agent import createAgent
import threading

from bulletarm_baselines.equi_rl.utils.torch_utils import ExpertTransition, normalizeTransition, augmentBuffer,augmentTransitionSE2, augmentBatch, augmentBatchNx, augmentBatchVec

from bulletarm_baselines.logger.baseline_logger import BaselineLogger

def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def train_step(agent, replay_buffer,n_training_steps):
    batch = replay_buffer.sample(batch_size)  
    if obs_type == 'pixel':
        if not no_augment:
            batch = augmentBatch(batch, aug_n=equi_n, aug_t="se2")
            

    if algorithm == 'cql' or algorithm == 'equi_cql' or algorithm == 'equi_cql_actor' or algorithm == 'equi_cql_critic' or algorithm == 'equi_cql_2' or algorithm == 'cql_2' or algorithm == 'equi_cql_critic_2' or algorithm == 'equi_cql_actor_2':
        qf1_loss, qf2_loss, policy_loss, alpha_loss, alpha_tlogs, td_error, ind_qf1_loss, ind_qf2_loss, cql1_scaled_loss, cql2_scaled_loss,avg_q_values_1, avg_q_values_2 = agent.update(batch)

        wandb.log({"Total QF1 Loss": qf1_loss, "Total QF2 Loss": qf2_loss, "Policy Loss": policy_loss, "Alpha Loss": alpha_loss, "Alpha Tlogs": alpha_tlogs, "TD Error": td_error, "CQL1 Scaled Loss": cql1_scaled_loss, "CQL2 Scaled Loss": cql2_scaled_loss,
            "QF1 Loss": ind_qf1_loss, "QF2 Loss": ind_qf2_loss,"Average Q Values 1": avg_q_values_1, "Average Q Values 2": avg_q_values_2}, step=n_training_steps)
    
    elif algorithm == 'iql' or algorithm == 'equi_iql' or algorithm == 'equi_iql_critic' or algorithm == 'equi_iql_critic_value' or algorithm == 'equi_iql_critic_Q' or algorithm == 'equi_iql_actor':
        qf1_loss, qf2_loss, policy_loss, avg_q_values_1, avg_q_values_2, vf_loss, avg_vf_values, avg_vf_target_values, avg_log_pis,avg_stds = agent.update(batch) 
        wandb.log({"Total QF1 Loss": qf1_loss, "Total QF2 Loss": qf2_loss, "Policy Loss": policy_loss, "Average Q Values 1": avg_q_values_1, "Average Q Values 2": avg_q_values_2,
                   "Value Function Loss": vf_loss, "Average Value Function Values": avg_vf_values, "Average Value Function Target Values": avg_vf_target_values, "Average Log Pis": avg_log_pis,"Average Stds": avg_stds}, step=n_training_steps)
    
    elif algorithm == 'equi_iql_cql' or algorithm == 'iql_cql':
        qf1_loss, qf2_loss, policy_loss, avg_q_values_1, avg_q_values_2, vf_loss, avg_vf_values, avg_vf_target_values, avg_log_pis, avg_stds, avg_dataset_q_values_1, avg_dataset_q_values_2 = agent.update(batch) 
        wandb.log({"Total QF1 Loss": qf1_loss, "Total QF2 Loss": qf2_loss, "Policy Loss": policy_loss, "Average Q Values 1": avg_q_values_1, "Average Q Values 2": avg_q_values_2,
                   "Value Function Loss": vf_loss, "Average Value Function Values": avg_vf_values, "Average Value Function Target Values": avg_vf_target_values, "Average Log Pis": avg_log_pis,"Average Stds": avg_stds,\
                    "Average Dataset Q Values 1": avg_dataset_q_values_1, "Average Dataset Q Values 2": avg_dataset_q_values_2}, step=n_training_steps)

    elif algorithm == 'bcfd' or algorithm == 'equi_bcfd':
      policy_loss = agent.update(batch)

      wandb.log({"Policy Loss": policy_loss}, step=n_training_steps)          



    if n_training_steps % target_update_freq == 0:
        agent.updateTarget()



def evaluate_dummy(envs, agent,result_list):
    states, obs = envs.reset()
    evaled = 0
    temp_reward = [[] for _ in range(num_eval_processes)]
    eval_rewards = []
    
    while evaled < 1:
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
        


    eval_rewards_numpy = np.array(eval_rewards)
    mean_reward = np.mean(eval_rewards_numpy)
    envs.close()


def evaluate(envs, agent_eval,result_list,highest_eval_reward_list,file_name,algorithm,num_training_steps):
    states, obs = envs.reset()
    evaled = 0
    temp_reward = [[] for _ in range(num_eval_processes)]
    eval_rewards = []
    
    if not no_bar:
        eval_bar = tqdm(total=num_eval_episodes)
    
    while evaled < num_eval_episodes:
        actions_star_idx, actions_star = agent_eval.getGreedyActions(states, obs)
        # action,_,_  = agent_eval.actor.sample(obs)
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
    result_list[0] = mean_reward

    if highest_eval_reward_list.qsize() < 3:
        try:
            filename = "models/" + file_name[:-1] + "/" + algorithm +"/model_" + str(num_training_steps) + "_" + str(result_list[0])  
            agent_eval.saveModel(filename)
            highest_eval_reward_list.put((result_list[0],filename))
        except:
            try:
                os.makedirs("models/" + file_name[:-1])
            except:
                pass
            try:
                os.makedirs("models/" + file_name[:-1] + "/" + algorithm)
            except:
                pass
            filename = "models/" + file_name[:-1] + "/" + algorithm + "/model_" + str(num_training_steps) + "_" + str(result_list[0])
            agent_eval.saveModel(filename)
            highest_eval_reward_list.put((result_list[0],filename))
    else:
        filename = "models/" + file_name[:-1] + "/" + algorithm + "/model_" + str(num_training_steps) + "_" + str(result_list[0])
        highest_eval_reward_list.put((result_list[0],filename))

        _, lowest_reward_model_filename = highest_eval_reward_list.get()
        # agent_eval.saveModel(filename)

        if filename != lowest_reward_model_filename:
            os.remove(lowest_reward_model_filename+"_0.pt")
            os.remove(lowest_reward_model_filename+"_1.pt")
            agent_eval.saveModel(filename)

    envs.close()




def train():
    highest_eval_reward = PriorityQueue()

    eval_thread = None
    start_time = time.time()
    if seed is not None:
        set_seed(seed)
    # setup env
    print('creating envs')
    envs = EnvWrapper(num_processes, env, env_config, planner_config)

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
            "Task": "Offline RL",           
        },
    )


    

    hyper_parameters['model_shape'] = agent.getModelStr()

    #list all the datasets in the datasets folder
    parent_dir = "datasets/"
    dataset_list = os.listdir(parent_dir)

    replay_buffer = QLearningBuffer(buffer_size)
    replay_buffer_expert = QLearningBuffer(buffer_size)
    replay_buffer_expert_actual = QLearningBufferExpert(buffer_size)
    
    if data_collection_type == 'optimal':
        print("Loading: {}".format("datasets/" + env + '_expert_buffer_'+str(data_expert_demos) + "_" + str(seed)))
        file_name = env + '_expert_buffer_'+str(data_expert_demos) 
        
        if obs_type != 'pixel':
            replay_buffer.loadBuffer("datasets/" + env + '_expert_buffer_'+str(data_expert_demos) + "_" + "vector" + "_" + str(seed) +'.pkl')
            file_name = env + '_expert_buffer_'+str(data_expert_demos) + "_" + "vector" + "_" + "seed" + str(seed) + "_" 
        else:
            replay_buffer.loadBuffer("datasets/" + env + '_expert_buffer_'+str(data_expert_demos) + "_" + str(seed) +'.pkl')
            file_name = env + '_expert_buffer_'+str(data_expert_demos) + "_" + "seed" + str(seed) + "_"


    elif data_collection_type == 'mix':
        if obs_type != 'pixel':
                file_name = env + '_sub_optimal_buffer_'+str(data_reward_limit) +"_" + "vector" + "_"
        else: 
            file_name = env + '_sub_optimal_buffer_'+str(data_reward_limit) +"_"
        if not data_collection_transitions: 
            for dataset in dataset_list:
                if file_name in dataset and seed == int(dataset[-5]) and data_demos == int(dataset[-6 - len(str(data_demos)) : -6]):
                    print("Loading: {}".format("datasets/" + dataset))
                    replay_buffer.loadBuffer("datasets/" + dataset)
                    print("Buffer Loaded, Size: {}".format(len(replay_buffer)))
                    break
        else:
            for dataset in dataset_list:
                if file_name in dataset and seed == int(dataset[-5]) and offline_buffer_size == int(dataset[-6 - len(str(offline_buffer_size)) : -6]):
                    print("Loading: {}".format("datasets/" + dataset))
                    replay_buffer.loadBuffer("datasets/" + dataset)
                    print("Buffer Loaded, Size: {}".format(len(replay_buffer)))
                    break
        
        print("Loading: {}".format("datasets/" + env + '_expert_buffer_'+str(data_expert_demos)))
        file_name = env + '_expert_buffer_'+str(data_expert_demos)
        replay_buffer_expert.loadBuffer("datasets/" + env + '_expert_buffer_'+str(data_expert_demos) + "_" + str(seed) +'.pkl')
        print("Expert Buffer Loaded, Size: {}".format(len(replay_buffer_expert)))

        for i in range(len(replay_buffer_expert)):
            replay_buffer_expert_actual.add(replay_buffer_expert[i])
        print("New Expert Buffer Loaded, Size: {}".format(len(replay_buffer)))
        

        replay_buffer_expert_actual._next_idx = len(replay_buffer_expert_actual)
        replay_buffer_expert_actual._max_size = buffer_size  + len(replay_buffer_expert_actual)

        for i in range(len(replay_buffer)):
            replay_buffer_expert_actual.add(replay_buffer[i])

        replay_buffer = replay_buffer_expert_actual

        print("Mix Buffer Loaded, Size: {}".format(len(replay_buffer)))
        file_name = env + '_mix_buffer_'+str(data_reward_limit) + "_demos_" + str(data_expert_demos) + "_"


    
    elif data_collection_type == 'sub_optimal':
        if not data_collection_policy_bc:
            if obs_type != 'pixel':
                file_name = env + '_sub_optimal_buffer_'+str(data_reward_limit) +"_" + "vector" + "_"
            else: 
                file_name = env + '_sub_optimal_buffer_'+str(data_reward_limit) +"_"
        else:
            if obs_type != 'pixel':
                    file_name = env + '_bc_sub_optimal_buffer_'+str(data_reward_limit) +"_" + "vector" + "_"
            else: 
                file_name = env + '_bc_sub_optimal_buffer_'+str(data_reward_limit) +"_"

        if not data_collection_transitions: 
            for dataset in dataset_list:
                if file_name in dataset and seed == int(dataset[-5]) and data_demos == int(dataset[-6 - len(str(data_demos)) : -6]):
                    print("Number of Demos:", int(dataset[-6 - len(str(data_demos)) : -6]))
                    print("Loading: {}".format("datasets/" + dataset))
                    replay_buffer.loadBuffer("datasets/" + dataset)
                    print("Buffer Loaded, Size: {}".format(len(replay_buffer)))
                    break
        else:
            for dataset in dataset_list:
                if file_name in dataset and seed == int(dataset[-5]) and offline_buffer_size == int(dataset[-6 - len(str(offline_buffer_size)) : -6]):
                    print("Loading: {}".format("datasets/" + dataset))
                    replay_buffer.loadBuffer("datasets/" + dataset)
                    print("Buffer Loaded, Size: {}".format(len(replay_buffer)))
                    break
    
    
    


    timer_start = time.time()
    num_training_steps = 0
    result = [None]
    if not no_bar:
        pbar = tqdm(total=max_train_step)
        pbar.set_description('Episodes:0; Time: 0.0')

    while num_training_steps < max_train_step:
        num_training_steps += 1

        
        for training_iter in range(training_iters):
            train_step(agent, replay_buffer,num_training_steps)

        
        if (time.time() - start_time)/3600 > time_limit:
            break

        if not no_bar:
            timer_final = time.time()

            description = 'Action Step:{}; Time:{:.03f}'.format(
                num_training_steps, timer_final - timer_start)
            
            pbar.set_description(description)
            timer_start = timer_final
            pbar.update(num_training_steps-pbar.n)

        if num_training_steps > 0 and eval_freq > 0 and num_training_steps % eval_freq == 0:
            if eval_thread is not None:
                eval_thread.join()
            eval_envs = EnvWrapper(num_eval_processes, env, env_config, planner_config)
            eval_agent.copyNetworksFrom(agent)
            evaluate_dummy(eval_envs, eval_agent,result)
            eval_envs = EnvWrapper(num_eval_processes, env, env_config, planner_config)
            eval_thread = threading.Thread(target=evaluate, args=(eval_envs, eval_agent,result,highest_eval_reward,file_name,algorithm,num_training_steps))
            eval_thread.start()

            if result[0] != None:
                wandb.log({"Evaluated Reward": result[0]}, step=num_training_steps)
            print("Highest Evaluated Reward: {}".format(highest_eval_reward.queue))

                    

            
    envs.close()
    print('training finished')
    if not no_bar:
        pbar.close()

if __name__ == '__main__':
    train()