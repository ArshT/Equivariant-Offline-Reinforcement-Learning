from bulletarm_baselines.equi_rl.agents.a2c_base import A2CBase
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from bulletarm_baselines.equi_rl.utils.parameters import heightmap_size, crop_size
from bulletarm_baselines.equi_rl.utils.torch_utils import centerCrop
import math

class IQL_CQL_SAC(A2CBase):
    """
    SAC agent class
    Part of the code for this class is referenced from https://github.com/pranz24/pytorch-soft-actor-critic
    """
    def __init__(self,quantile,beta,lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/16, n_a=5, tau=0.001,
                 policy_type='gaussian', target_update_interval=1, automatic_entropy_tuning=False,
                 obs_type='pixel',use_one_channel=False, iql_cql_alpha=1):
        super().__init__(lr, gamma, device, dx, dy, dz, dr, n_a, tau)
        self.policy_type = policy_type
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.obs_type = obs_type
        self.quantile = quantile
        self.beta = beta
        self.use_one_channel = use_one_channel

        self.num_update = 0

        self.goal_conditioned = False

        self.bc_alpha = iql_cql_alpha


    
    def initNetwork(self, actor, critic,value_fn,initialize_target=True):
        """
        Initialize networks
        :param actor: actor network
        :param critic: critic network
        :param initialize_target: whether to create target networks
        """
        self.actor = actor
        self.critic = critic
        self.value_fn = value_fn
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr[0])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr[1])
        self.value_fn_optimizer = torch.optim.Adam(self.value_fn.parameters(), lr=self.lr[0])

        if initialize_target:
            self.critic_target = deepcopy(critic)
            self.target_networks.append(self.critic_target)

            self.value_fn_target = deepcopy(value_fn)
            self.target_networks.append(self.value_fn_target)

        self.networks.append(self.actor)
        self.networks.append(self.critic)
        self.optimizers.append(self.actor_optimizer)
        self.optimizers.append(self.critic_optimizer)

        self.networks.append(self.value_fn)
        self.optimizers.append(self.value_fn_optimizer)



    def getSaveState(self):
        """
        Get the save state for checkpointing. Include network states, target network states, and optimizer states
        :return: the saving state dictionary
        """
        state = super().getSaveState()
        state['alpha'] = self.alpha
        state['log_alpha'] = self.log_alpha
        state['alpha_optimizer'] = self.alpha_optim.state_dict()
        return state

    def loadFromState(self, save_state):
        """
        Load from a save_state
        :param save_state: the loading state dictionary
        """
        super().loadFromState(save_state)
        self.alpha = save_state['alpha']
        self.log_alpha = torch.tensor(np.log(self.alpha.item()), requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=1e-3)
        self.alpha_optim.load_state_dict(save_state['alpha_optimizer'])




    def targetSoftUpdate(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = self.tau

        for t_param, l_param in zip(
                self.critic_target.parameters(), self.critic.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        
        
        for t_param, l_param in zip(
                self.value_fn_target.parameters(), self.value_fn.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)
        




    def getEGreedyActions(self, state, obs, eps, goals=None):
        """
        Get stochastic behavior policy's action. Note that this function is called getEGreedyActions, but it uses SAC's
        gaussian distribution to sample actions instead of e-greedy
        :param state: gripper holding state
        :param obs: observation
        :param eps: epsilon (not used)
        :return: unscaled_actions (in range (-1, 1)), actions (in true scale)
        """
        return self.getSACAction(state, obs, evaluate=False,goals=goals)

    def getGreedyActions(self, state, obs,goals=None):
        """
        Get greedy actions
        :param state: gripper holding state
        :param obs: observation
        :return: unscaled_actions (in range (-1, 1)), actions (in true scale)
        """
        return self.getSACAction(state, obs, evaluate=True,goals=goals)

    def getSACAction(self, state, obs,evaluate, goals=None):
        """
        Get SAC action (greedy or sampled from gaussian, based on evaluate flag)
        :param state: gripper holding state
        :param obs: observation
        :param evaluate: if evaluate==True, return greedy action. Otherwise return action sampled from gaussian
        :return: unscaled_actions (in range (-1, 1)), actions (in true scale)
        """
        with torch.no_grad():
            if self.obs_type is 'pixel':
                if not self.use_one_channel:
                    state_tile = state.reshape(state.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])
                    obs = torch.cat([obs, state_tile], dim=1).to(self.device)
                    if heightmap_size > crop_size:
                        obs = centerCrop(obs, out=crop_size)
                else:
                    if heightmap_size > crop_size:
                        obs = centerCrop(obs, out=crop_size)
                    
                    obs = obs.to(self.device)
            else:
                obs = obs.to(self.device)

            if evaluate is False:
                if goals is not None:
                    action, _, _ = self.actor.sample(obs,goals)
                else:
                    action, _, _ = self.actor.sample(obs)
            else:
                if goals is not None:
                    _, _, action = self.actor.sample(obs,goals)
                else:
                    _, _, action = self.actor.sample(obs)
            action = action.to('cpu')
            return self.decodeActions(*[action[:, i] for i in range(self.n_a)])

    
    
    def _loadLossCalcDict(self):
        """
        get the loaded batch data in self.loss_calc_dict
        :return: batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts
        """
        try:
            batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts, goals = super()._loadLossCalcDict()
            self.goal_conditioned = True
        except:
            batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = super()._loadLossCalcDict()

        if self.obs_type is 'pixel':
            if not self.use_one_channel:
                obs = torch.cat([obs, states.reshape(states.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])], dim=1)
                next_obs = torch.cat([next_obs, next_states.reshape(next_states.size(0), 1, 1, 1).repeat(1, 1, next_obs.shape[2], next_obs.shape[3])], dim=1)
            else:
                pass

        
        try:
            return batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts, goals
        except:
            return batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts


    

    def calcLosses(self):
        """
        Calculate critic loss
        :return: q1 loss, q2 loss, td error
        """
        try:
            batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts, goals = self._loadLossCalcDict()
            self.goal_conditioned = True
        except:
            batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()



        '''
        QF Losses
        '''
        if not self.goal_conditioned:
            qf1, qf2 = self.critic(obs, action)  # Two Q-functions to mitigate positive bias in the policy improvement step
        else:
            qf1, qf2 = self.critic(obs, action, goals)
        qf1 = qf1.reshape(batch_size)
        qf2 = qf2.reshape(batch_size)

        if not self.goal_conditioned:
            target_vf_pred = self.value_fn_target(next_obs).detach()
        else:
            target_vf_pred = self.value_fn_target(next_obs, goals).detach()
        target_vf_pred = target_vf_pred.reshape(batch_size)

        q_target = rewards + non_final_masks * self.gamma * target_vf_pred
        q_target = q_target.detach()

        qf1_loss = F.mse_loss(qf1, q_target)
        qf2_loss = F.mse_loss(qf2, q_target)


        '''
        Value Function Loss
        '''
        if not self.goal_conditioned:
            qf1_target, qf2_target = self.critic_target(obs,action)
        else:
            qf1_target, qf2_target = self.critic_target(obs, action, goals)
        q_pred = torch.min(qf1_target, qf2_target).detach()

        if not self.goal_conditioned:
            vf_pred = self.value_fn(obs)
        else:
            vf_pred = self.value_fn(obs, goals)

        vf_err = vf_pred - q_pred
        vf_sign = (vf_err > 0).float()
        vf_weight = (1 - vf_sign) * self.quantile + vf_sign * (1 - self.quantile)
        vf_loss = (vf_weight * (vf_err ** 2)).mean()




        '''
        Policy Loss
        '''
        if not self.goal_conditioned:
            log_pi_bc, _, std = self.actor.get_logprobs(obs, action)
        else:
            log_pi_bc, _, std = self.actor.get_logprobs(obs, action, goals)
        
        ######################################################################
        ## Advantage-Weighted Regression
        # adv = q_pred - vf_pred
        # exp_adv = torch.exp(adv / self.beta)
        
        # weights = exp_adv[:, 0].detach()
        # policy_loss = (-log_pi * weights).mean()
        ######################################################################
        
        if self.goal_conditioned:
            pi, _, _ = self.actor.sample(obs,goals)
        else:
            pi, _, _ = self.actor.sample(obs)


        if self.goal_conditioned:
            qf1_pi, qf2_pi = self.critic(obs, pi,goals)
        else:
            qf1_pi, qf2_pi = self.critic(obs, pi)


        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = -((self.bc_alpha * log_pi_bc) + min_qf_pi).mean()

        avg_q_values_1 = qf1_pi.mean().item()
        avg_q_values_2 = qf2_pi.mean().item()
        avg_vf_values = vf_pred.mean().item()
        avg_vf_target_values = target_vf_pred.mean().item()
        avg_log_pi_bc = (log_pi_bc).mean().item()
        avg_std = std.mean().item()

        avg_dataset_q_values_1 = qf1.mean().item()
        avg_dataset_q_values_2 = qf2.mean().item()
                   
        return qf1_loss, qf2_loss,policy_loss,vf_loss, avg_q_values_1, avg_q_values_2, avg_vf_values, avg_vf_target_values, avg_log_pi_bc, avg_std, avg_dataset_q_values_1, avg_dataset_q_values_2 

    
    
    def update(self, batch):
        """
        Perform a training step
        :param batch: the sampled minibatch
        :return: loss
        """
        self._loadBatchToDevice(batch)

        qf1_loss, qf2_loss, policy_loss, vf_loss, avg_q_values_1, avg_q_values_2, avg_vf_values, avg_vf_target_values, avg_log_pis, avg_stds, avg_dataset_q_values_1, avg_dataset_q_values_2 = self.calcLosses()
        qf_loss = qf1_loss + qf2_loss

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        self.value_fn_optimizer.zero_grad()
        vf_loss.backward()
        self.value_fn_optimizer.step()
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()



        self.num_update += 1
        if self.num_update % self.target_update_interval == 0:
            self.targetSoftUpdate()

        self.loss_calc_dict = {}

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), avg_q_values_1, avg_q_values_2, vf_loss.item(), avg_vf_values, avg_vf_target_values, avg_log_pis, avg_stds, avg_dataset_q_values_1, avg_dataset_q_values_2
