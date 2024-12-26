from bulletarm_baselines.equi_rl.utils.parameters import *
from bulletarm_baselines.equi_rl.agents.dqn_agent_com import DQNAgentCom
from bulletarm_baselines.equi_rl.agents.dqn_agent_com_drq import DQNAgentComDrQ
from bulletarm_baselines.equi_rl.agents.curl_dqn_com import CURLDQNCom
from bulletarm_baselines.equi_rl.networks.dqn_net import CNNCom
from bulletarm_baselines.equi_rl.networks.equivariant_dqn_net import EquivariantCNNCom

from bulletarm_baselines.equi_rl.agents.sac import SAC
from bulletarm_baselines.equi_rl.agents.sac_new import SAC_2
from bulletarm_baselines.equi_rl.agents.sacfd import SACfD
from bulletarm_baselines.equi_rl.agents.bcfd import BCfD
from bulletarm_baselines.equi_rl.agents.curl_sac import CURLSAC
from bulletarm_baselines.equi_rl.agents.curl_sacfd import CURLSACfD
from bulletarm_baselines.equi_rl.agents.sac_drq import SACDrQ
from bulletarm_baselines.equi_rl.agents.sacfd_drq import SACfDDrQ
from bulletarm_baselines.equi_rl.agents.cql_sac import CQL_SAC
# from bulletarm_baselines.equi_rl.agents.cql_sac_new import CQL_SAC_2
from bulletarm_baselines.equi_rl.agents.cql_sac_2 import CQL_SAC_2
from bulletarm_baselines.equi_rl.agents.iql_sac import IQL_SAC
from bulletarm_baselines.equi_rl.agents.iql_cql_sac import IQL_CQL_SAC

from bulletarm_baselines.equi_rl.networks.sac_net import SACCritic, SACGaussianPolicy, SACIQLPolicy, SACIQLValueCritic,SACIQLValueCritic, SACGaussianPolicy_2_Objects,SACCritic_2_Objects
from bulletarm_baselines.equi_rl.networks.sac_net import SACCriticGoal, SACIQLValueCriticGoal, SACIQLPolicyGoal, SACGaussianPolicyGoal
from bulletarm_baselines.equi_rl.networks.equivariant_sac_net import EquivariantSACActor, EquivariantSACCritic, EquivariantSACActorDihedral, EquivariantSACCriticDihedral, EquivariantSACValueCritic,EquivariantSACValueCritic2, EquivariantIQLActor, EquivariantSACActor_2_Objects, EquivariantSACCritic_2_Objects, EquivariantSACActor_1_Object, EquivariantSACCritic_1_Object\
                                                                    ,EquivariantSACCValueCritic_2_Objects
from bulletarm_baselines.equi_rl.networks.equivariant_sac_net import EquivariantIQLActorGoal, EquivariantSACCriticGoal, EquivariantSACValueCriticGoal, EquivariantSACValueCritic2Goal, EquivariantSACActorGoal
from bulletarm_baselines.equi_rl.networks.equivariant_sac_net import EquivariantSACCriticCQL, EquivariantSACActorCQL
from bulletarm_baselines.equi_rl.networks.sac_net import SACCriticCQL, SACGaussianPolicyCQL
from bulletarm_baselines.equi_rl.networks.curl_sac_net import CURLSACCritic, CURLSACGaussianPolicy, CURLSACEncoderOri, CURLSACEncoder
from bulletarm_baselines.equi_rl.networks.dqn_net import DQNComCURL, DQNComCURLOri

import torch
import cv2 as cv



def test_equi_both_critic(equi_critic):
    print("Testing Equi Both Critic")

    obs_shape = (2, 129, 129)
    action_dim = 5
    n_hidden = 128
    N = 8
    enc_id = 1

    obs = torch.zeros(1, *obs_shape).to("cuda")
    obs[:,:,10:20,10:20] = 1
    
    act = torch.tensor([1,1,1,1,1]).to("cuda")
    act = act.reshape(1, -1).float()
    print("Observation Shape:", obs.shape)
    print("Action Shape:", act.shape)
    print()

    q1, q2 = equi_critic(obs, act)
    print(q1.item(), q2.item())

    g_obs = torch.zeros(1, *obs_shape).to("cuda")
    g_obs[:,:,-20:-10,-20:-10] = 1

    g_act = torch.tensor([1,-1,-1,1,1]).to("cuda")
    g_act = g_act.reshape(1, -1).float()
    print("Observation Shape:", g_obs.shape)
    print("Action Shape:", g_act.shape)

    g_q1, g_q2 = equi_critic(g_obs, g_act)
    print(g_q1.item(), g_q2.item())

    check_q1,check_q2 = equi_critic(g_obs, act)
    print(check_q1.item(), check_q2.item())
    print()


    obs_new_2 = obs.clone().squeeze(0).detach().cpu().numpy()
    (h, w) = obs_new_2[0].shape[:2]
    center = (int(w / 2), int(h / 2))
    M = cv.getRotationMatrix2D(center, 180, 1.0)
    obs_new_2 = cv.warpAffine(obs_new_2[0], M, (w, h)) 
    g_obs_new_2_0 = torch.tensor(obs_new_2).unsqueeze(0).to("cuda")
    g_obs_new_2 = obs.clone().detach()
    g_obs_new_2[:,0,:,:] = g_obs_new_2_0
    g_obs_new_2 = g_obs_new_2.float().to("cuda")

    g_2_q1, g_2_q2 = equi_critic(g_obs, g_act)
    print(g_2_q1.item(), g_2_q2.item())



def createAgent(test=False,goal=False):

    if use_one_channel:
        obs_channel = 1
    else:
        obs_channel = 2


    if load_sub is not None or load_model_pre is not None or test:
        initialize = False
    else:
        initialize = True
    n_p = 2
    if not random_orientation:
        n_theta = 1
    else:
        n_theta = 3

    print('initializing agent')
    print("Initialize:",initialize)
    
    

    # setup agent
    if alg in ['dqn_com', 'dqn_com_drq']:
        if alg == 'dqn_com':
            agent = DQNAgentCom(lr=lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot, n_p=n_p, n_theta=n_theta)
        elif alg == 'dqn_com_drq':
            agent = DQNAgentComDrQ(lr=lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot, n_p=n_p,
                                   n_theta=n_theta)
        else:
            raise NotImplementedError
        if model == 'cnn':
            net = CNNCom((obs_channel, crop_size, crop_size), n_p=n_p, n_theta=n_theta).to(device)
        elif model == 'equi':
            net = EquivariantCNNCom(n_p=n_p, n_theta=n_theta, initialize=initialize).to(device)
        else:
            raise NotImplementedError
        agent.initNetwork(net, initialize_target=not test)

    elif alg in ['curl_dqn_com']:
        if alg == 'curl_dqn_com':
            agent = CURLDQNCom(lr=lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot, n_p=n_p,
                               n_theta=n_theta, crop_size=crop_size)
        else:
            raise NotImplementedError
        if model == 'cnn':
            net = DQNComCURL((obs_channel, crop_size, crop_size), n_p, n_theta).to(device)
        # network from curl paper
        elif model == 'cnn_curl':
            net = DQNComCURLOri((obs_channel, crop_size, crop_size), n_p, n_theta).to(device)
        else:
            raise NotImplementedError
        agent.initNetwork(net)

    elif alg in ['sac', 'sacfd', 'sacfd_mean', 'sac_drq', 'sacfd_drq','bcfd','cql_sac','cql_sac_2','sac_2','iql_sac','iql_cql_sac']:
        sac_lr = (actor_lr, critic_lr)
        if alg == 'sac':
            agent = SAC(lr=sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                        n_a=len(action_sequence), tau=tau, alpha=init_temp, policy_type='gaussian',
                        target_update_interval=1, automatic_entropy_tuning=True, obs_type=obs_type)
        elif alg == 'sac_2':
            agent = SAC_2(lr=sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                        n_a=len(action_sequence), tau=tau, alpha=init_temp, policy_type='gaussian',
                        target_update_interval=1, automatic_entropy_tuning=True, obs_type=obs_type)
        elif alg == 'sacfd':
            agent = SACfD(lr=sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                          n_a=len(action_sequence), tau=tau, alpha=init_temp, policy_type='gaussian',
                          target_update_interval=1, automatic_entropy_tuning=True, obs_type=obs_type,
                          demon_w=demon_w)
        elif alg == 'bcfd':
            agent = BCfD(lr=sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                          n_a=len(action_sequence), tau=tau, alpha=init_temp, policy_type='gaussian',
                          target_update_interval=1, automatic_entropy_tuning=True, obs_type=obs_type,
                          demon_w=demon_w)
        elif alg == 'sacfd_mean':
            agent = SACfD(lr=sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                          n_a=len(action_sequence), tau=tau, alpha=init_temp, policy_type='gaussian',
                          target_update_interval=1, automatic_entropy_tuning=True, obs_type=obs_type,
                          demon_w=demon_w, demon_l='mean')
        elif alg == 'sac_drq':
            agent = SACDrQ(lr=sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                           n_a=len(action_sequence), tau=tau, alpha=init_temp, policy_type='gaussian',
                           target_update_interval=1, automatic_entropy_tuning=True, obs_type=obs_type)
        elif alg == 'sacfd_drq':
            agent = SACfDDrQ(lr=sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                             n_a=len(action_sequence), tau=tau, alpha=init_temp, policy_type='gaussian',
                             target_update_interval=1, automatic_entropy_tuning=True, obs_type=obs_type,
                             demon_w=demon_w)
        elif alg == 'cql_sac':
            agent = CQL_SAC(lr=sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                            n_a=len(action_sequence), tau=tau, alpha=init_temp, policy_type='gaussian',
                            target_update_interval=1, automatic_entropy_tuning=True, obs_type=obs_type,
                            temp=cql_temperature, cql_weight=cql_weight,with_lagrange=with_lagrange,
                            target_action_gap=target_action_gap,use_one_channel=use_one_channel)
        elif alg == 'cql_sac_2':
            agent = CQL_SAC_2(lr=sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                            n_a=len(action_sequence), tau=tau, alpha=init_temp, policy_type='gaussian',
                            target_update_interval=1, automatic_entropy_tuning=True, obs_type=obs_type,
                            temp=cql_temperature, cql_weight=cql_weight,with_lagrange=with_lagrange,
                            target_action_gap=target_action_gap)
        elif alg == 'iql_sac':
            agent = IQL_SAC(lr=sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                            n_a=len(action_sequence), tau=tau, policy_type='gaussian',
                            target_update_interval=1, automatic_entropy_tuning=True, obs_type=obs_type,
                            quantile=iql_quantile, beta=iql_beta, use_one_channel=use_one_channel)
        elif alg == 'iql_sac':
            agent = IQL_SAC(lr=sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                            n_a=len(action_sequence), tau=tau, policy_type='gaussian',
                            target_update_interval=1, automatic_entropy_tuning=True, obs_type=obs_type,
                            quantile=iql_quantile, beta=iql_beta, use_one_channel=use_one_channel)
        elif alg == 'iql_cql_sac':
            agent = IQL_CQL_SAC(lr=sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                            n_a=len(action_sequence), tau=tau, policy_type='gaussian',
                            target_update_interval=1, automatic_entropy_tuning=True, obs_type=obs_type,
                            quantile=iql_quantile, beta=iql_beta, use_one_channel=use_one_channel, iql_cql_alpha=iql_cql_alpha)
        else:
            raise NotImplementedError
        # pixel observation
        if obs_type == 'pixel':
            if model == 'cnn':

                if alg == 'iql_sac' or alg == 'iql_cql_sac':
                    if not goal:
                        actor = SACIQLPolicy((obs_channel, crop_size, crop_size), len(action_sequence)).to(device)
                        value_fn = SACIQLValueCritic((obs_channel, crop_size, crop_size)).to(device)
                    else:
                        actor = SACIQLPolicyGoal((obs_channel, crop_size, crop_size), len(action_sequence), n_tasks=goal_num_tasks).to(device)
                        value_fn = SACIQLValueCriticGoal((obs_channel, crop_size, crop_size), n_tasks=goal_num_tasks).to(device)
                else:
                    if not goal:
                        if alg == 'cql_sac_2':
                            actor = SACGaussianPolicyCQL((obs_channel, crop_size, crop_size), len(action_sequence)).to(device)
                        else:
                            actor = SACGaussianPolicy((obs_channel, crop_size, crop_size), len(action_sequence)).to(device)
                    else:
                        actor = SACGaussianPolicyGoal((obs_channel, crop_size, crop_size), len(action_sequence), n_tasks=goal_num_tasks).to(device)
                
                if not goal:
                    if alg == 'cql_sac_2':
                        critic = SACCriticCQL((obs_channel, crop_size, crop_size), len(action_sequence)).to(device)
                    else:
                        critic = SACCritic((obs_channel, crop_size, crop_size), len(action_sequence)).to(device)
                else:
                    critic = SACCriticGoal((obs_channel, crop_size, crop_size), len(action_sequence), n_tasks=goal_num_tasks).to(device)
            
            elif model == 'equi_actor':
                if alg == 'iql_sac' or alg == 'iql_cql_sac':
                    actor = EquivariantIQLActor((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
                    value_fn = SACIQLValueCritic((obs_channel, crop_size, crop_size)).to(device)
                else:
                    if alg == 'cql_sac_2':
                        actor = EquivariantSACActorCQL((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
                    else:
                        actor = EquivariantSACActor((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
                if alg == 'cql_sac_2':
                    critic = EquivariantSACCriticCQL((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
                else:
                    critic = SACCritic((obs_channel, crop_size, crop_size), len(action_sequence)).to(device)
            
            
            elif model == 'equi_critic':
                if alg == 'iql_sac' or alg == 'iql_cql_sac':
                    value_fn = EquivariantSACValueCritic2((obs_channel, crop_size, crop_size), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
                    actor = SACIQLPolicy((obs_channel, crop_size, crop_size), len(action_sequence)).to(device)
                else:
                    if alg == 'cql_sac_2':
                        actor = SACGaussianPolicyCQL((obs_channel, crop_size, crop_size), len(action_sequence)).to(device)
                    else:
                        actor = SACGaussianPolicy((obs_channel, crop_size, crop_size), len(action_sequence)).to(device)
                
                if alg == 'cql_sac_2':
                    critic = EquivariantSACCriticCQL((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
                else:
                    critic = EquivariantSACCritic((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)

            elif model == 'equi_critic_value':
                if alg == 'iql_sac' or alg == 'iql_cql_sac':
                    value_fn = EquivariantSACValueCritic2((obs_channel, crop_size, crop_size), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
                    actor = SACIQLPolicy((obs_channel, crop_size, crop_size), len(action_sequence)).to(device)
                critic = SACCritic((obs_channel, crop_size, crop_size), len(action_sequence)).to(device)
            
            elif model == 'equi_critic_Q':
                if alg == 'iql_sac' or alg == 'iql_cql_sac':
                    value_fn = SACIQLValueCritic((obs_channel, crop_size, crop_size)).to(device)
                    actor = SACIQLPolicy((obs_channel, crop_size, crop_size), len(action_sequence)).to(device)
                critic = EquivariantSACCritic((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
                
            
            elif model == 'equi_both':
                if alg == 'iql_sac' or alg == 'iql_cql_sac':
                    if not goal:
                        value_fn = EquivariantSACValueCritic2((obs_channel, crop_size, crop_size), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
                        actor = EquivariantIQLActor((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
                    else:
                        print("Num Tasks:", goal_num_tasks)
                        value_fn = EquivariantSACValueCritic2Goal((obs_channel, crop_size, crop_size), n_hidden=n_hidden, initialize=initialize, N=equi_n, n_tasks=goal_num_tasks).to(device)
                        actor = EquivariantIQLActorGoal((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n, n_tasks=goal_num_tasks).to(device)
                else:
                    if not goal:
                        if alg == 'cql_sac_2':
                            actor = EquivariantSACActorCQL((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
                        else:
                            actor = EquivariantSACActor((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
                    else:
                        print("Num Tasks:", goal_num_tasks)
                        actor = EquivariantSACActorGoal((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n, n_tasks=goal_num_tasks).to(device)
                
                if not goal:
                    if alg == 'cql_sac_2':
                        critic = EquivariantSACCriticCQL((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
                    else:
                        critic = EquivariantSACCritic((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
                else:
                    print("Num Tasks:", goal_num_tasks)
                    critic = EquivariantSACCriticGoal((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n, n_tasks=goal_num_tasks).to(device)

            elif model == 'equi_both_d':
              actor = EquivariantSACActorDihedral((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
              critic = EquivariantSACCriticDihedral((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
            else:
                raise NotImplementedError
        else:
            if model == 'equi_both':
                print("Using Equi MLP")
                if num_objects == 2:
                    print("Task with 2 objects")
                    actor = EquivariantSACActor_2_Objects((16,), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
                    critic = EquivariantSACCritic_2_Objects((16,), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
                elif num_objects == 1:
                    print("Task with 1 object")
                    actor = EquivariantSACActor_1_Object((16,), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
                    critic = EquivariantSACCritic_1_Object((16,), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
            elif model == 'cnn':
                if num_objects == 2:
                    actor = SACGaussianPolicy_2_Objects((16,), len(action_sequence)).to(device)
                    critic = SACCritic_2_Objects((16,), len(action_sequence)).to(device)
                elif num_objects == 1:
                    actor = SACGaussianPolicy_2_Objects((11,), len(action_sequence)).to(device)
                    critic = SACCritic_2_Objects((11,), len(action_sequence)).to(device)
            elif model == 'equi_critic':
                actor = SACGaussianPolicy_2_Objects((16,), len(action_sequence)).to(device)
                critic = EquivariantSACCritic_2_Objects((16,), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
            elif model == 'equi_actor':
                actor = EquivariantSACActor_2_Objects((16,), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
                critic = SACCritic_2_Objects((16,), len(action_sequence)).to(device)



        if alg == 'iql_sac' or alg == 'iql_cql_sac':
            agent.initNetwork(actor, critic, value_fn,not test)
        else:
            agent.initNetwork(actor, critic, not test)

    elif alg in ['curl_sac', 'curl_sacfd', 'curl_sacfd_mean']:
        curl_sac_lr = [actor_lr, critic_lr, lr, lr]
        if alg == 'curl_sac':
            agent = CURLSAC(lr=curl_sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot, n_a=len(action_sequence),
                            tau=tau, alpha=init_temp, policy_type='gaussian', target_update_interval=1, automatic_entropy_tuning=True,
                            crop_size=crop_size)
        elif alg == 'curl_sacfd':
            agent = CURLSACfD(lr=curl_sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                              n_a=len(action_sequence), tau=tau, alpha=init_temp, policy_type='gaussian',
                              target_update_interval=1, automatic_entropy_tuning=True, crop_size=crop_size,
                              demon_w=demon_w, demon_l='pi')
        elif alg == 'curl_sacfd_mean':
            agent = CURLSACfD(lr=curl_sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                              n_a=len(action_sequence), tau=tau, alpha=init_temp, policy_type='gaussian',
                              target_update_interval=1, automatic_entropy_tuning=True, crop_size=crop_size,
                              demon_w=demon_w, demon_l='mean')
        else:
            raise NotImplementedError
        if model == 'cnn':
            actor = CURLSACGaussianPolicy(CURLSACEncoder((obs_channel, crop_size, crop_size)).to(device), action_dim=len(action_sequence)).to(device)
            critic = CURLSACCritic(CURLSACEncoder((obs_channel, crop_size, crop_size)).to(device), action_dim=len(action_sequence)).to(device)
        # ferm paper network
        elif model == 'cnn_ferm':
            actor = CURLSACGaussianPolicy(CURLSACEncoderOri((obs_channel, crop_size, crop_size)).to(device),
                                          action_dim=len(action_sequence)).to(device)
            critic = CURLSACCritic(CURLSACEncoderOri((obs_channel, crop_size, crop_size)).to(device),
                                   action_dim=len(action_sequence)).to(device)
        else:
            raise NotImplementedError
        agent.initNetwork(actor, critic)
    else:
        raise NotImplementedError
    agent.aug = aug
    agent.aug_type = aug_type
    print('initialized agent')
    return agent