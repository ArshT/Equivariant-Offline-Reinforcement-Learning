from bulletarm_baselines.equi_rl.agents.a2c_base import A2CBase
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from bulletarm_baselines.equi_rl.utils.parameters import heightmap_size, crop_size
from bulletarm_baselines.equi_rl.utils.torch_utils import centerCrop
import math

class CQL_SAC_2(A2CBase):
    """
    SAC agent class
    Part of the code for this class is referenced from https://github.com/pranz24/pytorch-soft-actor-critic
    """
    def __init__(self, temp,with_lagrange,cql_weight,target_action_gap,lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/16, n_a=5, tau=0.001,
                 alpha=0.01, policy_type='gaussian', target_update_interval=1, automatic_entropy_tuning=False,
                 obs_type='pixel',use_one_channel=False):
        super().__init__(lr, gamma, device, dx, dy, dz, dr, n_a, tau)
        self.alpha = alpha
        self.policy_type = policy_type
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.obs_type = obs_type
        self.temp = temp
        self.with_lagrange = with_lagrange
        self.cql_weight = cql_weight
        self.target_action_gap = target_action_gap
        self.use_one_channel = use_one_channel

        self.goal_conditioned = False
        

        if self.policy_type == 'gaussian':
            if self.automatic_entropy_tuning is True:
                # self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.target_entropy = -n_a
                self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device)
                # self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=1e-4, betas=(0.5, 0.999))
                self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=1e-3)

        self.num_update = 0

    def initNetwork(self, actor, critic, initialize_target=True):
        """
        Initialize networks
        :param actor: actor network
        :param critic: critic network
        :param initialize_target: whether to create target networks
        """
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr[0])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr[1])
        if initialize_target:
            self.critic_target = deepcopy(critic)
            self.target_networks.append(self.critic_target)
        self.networks.append(self.actor)
        self.networks.append(self.critic)
        self.optimizers.append(self.actor_optimizer)
        self.optimizers.append(self.critic_optimizer)
        self.optimizers.append(self.alpha_optim)

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


    def getEGreedyActions(self, state, obs, eps,goals=None):
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

    def getSACAction(self, state, obs, evaluate,goals=None):
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
                    
                    obs = obs.to(self.device)
                else:
                    if heightmap_size > crop_size:
                        obs = centerCrop(obs, out=crop_size)
                    
                    obs = obs.to(self.device)
            else:
                obs = obs.to(self.device)

            if evaluate is False:
                try:
                    _, _, action = self.actor.sample(obs,goals)
                except:
                    action, _, _ = self.actor.sample(obs)
            else:
                try:
                    _, _, action = self.actor.sample(obs,goals)
                except:
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
            # stack state as the second channel of the obs
            if not self.use_one_channel:
                obs = torch.cat([obs, states.reshape(states.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])], dim=1)
                next_obs = torch.cat([next_obs, next_states.reshape(next_states.size(0), 1, 1, 1).repeat(1, 1, next_obs.shape[2], next_obs.shape[3])], dim=1)
            else:
                print("Yes")

        try:
            return batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts, goals
        except:
            return batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts

    
    
    
    
    def calcActorLoss(self):
        """
        Calculate actor loss
        :return: actor loss
        """
        try:
            batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts, goals = self._loadLossCalcDict()
            self.goal_conditioned = True
        except:
            batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()


        if self.goal_conditioned:
            pi, log_pi, mean = self.actor.sample(obs,goals)
        else:
            pi, log_pi, mean = self.actor.sample(obs)


        self.loss_calc_dict['pi'] = pi
        self.loss_calc_dict['mean'] = mean
        self.loss_calc_dict['log_pi'] = log_pi


        if self.goal_conditioned:
            qf1_pi, qf2_pi = self.critic(obs, pi,goals)
        else:
            qf1_pi, qf2_pi = self.critic(obs, pi)


        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        return policy_loss



    def calcCriticLoss(self):
        """
        Calculate critic loss
        :return: q1 loss, q2 loss, td error
        """
        try:
            batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts, goals = self._loadLossCalcDict()
            self.goal_conditioned = True
        except:
            batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()


        with torch.no_grad():
            if self.goal_conditioned:
                next_state_action, next_state_log_pi, _ = self.actor.sample(next_obs,goals)
            else:
                if len(next_obs.shape) == 2:
                    next_state_action, next_state_log_pi, _ = self.actor.sample(next_obs)
                else:
                    actor_next_obs_features = self.actor.forward_features(next_obs)
                    next_state_action, next_state_log_pi, _ = self.actor.sample_head(actor_next_obs_features)

            next_state_log_pi = next_state_log_pi.reshape(batch_size)

            if self.goal_conditioned:
                qf1_next_target, qf2_next_target = self.critic_target(next_obs, next_state_action,goals)
            else:
                if len(next_obs.shape) == 2:
                    qf1_next_target, qf2_next_target = self.critic_target(next_obs, next_state_action)
                else:
                    critic_next_obs_features = self.critic_target.forward_features(next_obs)
                    qf1_next_target, qf2_next_target = self.critic_target.forward_critic_head(critic_next_obs_features, next_state_action)


            qf1_next_target = qf1_next_target.reshape(batch_size)
            qf2_next_target = qf2_next_target.reshape(batch_size)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rewards + non_final_masks * self.gamma * min_qf_next_target

        
        if self.goal_conditioned:
            qf1, qf2 = self.critic(obs, action,goals)  # Two Q-functions to mitigate positive bias in the policy improvement step
        else:
            if len(obs.shape) == 2:
                qf1, qf2 = self.critic(obs, action)
            else:
                critic_obs_features = self.critic.forward_features(obs)
                qf1, qf2 = self.critic.forward_critic_head(critic_obs_features, action)  # Two Q-functions to mitigate positive bias in the policy improvement step

        qf1 = qf1.reshape(batch_size)
        qf2 = qf2.reshape(batch_size)

        if len(obs.shape) != 2:
            actor_obs_features = self.actor.forward_features(obs)
        
        # CQL addon
        random_actions = torch.FloatTensor(qf1.shape[0] * 10, action.shape[-1]).uniform_(-1, 1).to(self.device)
        num_repeat = int (random_actions.shape[0] / states.shape[0])

        if len(obs.shape) == 2:
            temp_obs = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
            temp_next_obs = next_obs.unsqueeze(1).repeat(1, num_repeat, 1).view(next_obs.shape[0] * num_repeat, next_obs.shape[1])
        elif len(critic_obs_features.shape) == 2:
            temp_critic_obs_features = critic_obs_features.unsqueeze(1).repeat(1, num_repeat, 1).view(critic_obs_features.shape[0] * num_repeat, critic_obs_features.shape[1])
        else:
            temp_critic_obs_features = critic_obs_features.unsqueeze(1).repeat(1, num_repeat, 1,1,1).view(critic_obs_features.shape[0] * num_repeat, critic_obs_features.shape[1],critic_obs_features.shape[2],critic_obs_features.shape[3])
        

        if len(actor_obs_features.shape) == 2:
            temp_actor_obs_features = actor_obs_features.unsqueeze(1).repeat(1, num_repeat, 1).view(actor_obs_features.shape[0] * num_repeat, actor_obs_features.shape[1])
            temp_actor_next_obs_features = actor_next_obs_features.unsqueeze(1).repeat(1, num_repeat, 1).view(actor_next_obs_features.shape[0] * num_repeat, actor_next_obs_features.shape[1])
        else:
            temp_actor_obs_features = actor_obs_features.unsqueeze(1).repeat(1, num_repeat, 1,1,1).view(actor_obs_features.shape[0] * num_repeat, actor_obs_features.shape[1],actor_obs_features.shape[2],actor_obs_features.shape[3])
            temp_actor_next_obs_features = actor_next_obs_features.unsqueeze(1).repeat(1, num_repeat, 1,1,1).view(actor_next_obs_features.shape[0] * num_repeat, actor_next_obs_features.shape[1],actor_next_obs_features.shape[2],actor_next_obs_features.shape[3])


        # if self.goal_conditioned:
        #     temp_goals = goals.unsqueeze(1).repeat(1, num_repeat, 1).view(goals.shape[0] * num_repeat, goals.shape[1])        


        if self.goal_conditioned:
            current_pi_values1, current_pi_values2  = self._compute_policy_values(temp_obs, temp_obs, goals=temp_goals)
            next_pi_values1, next_pi_values2 = self._compute_policy_values(temp_next_obs, temp_obs, goals=temp_goals)
        else:
            if len(obs.shape) == 2:
                current_pi_values1, current_pi_values2  = self._compute_policy_values(temp_obs, temp_obs)
                next_pi_values1, next_pi_values2 = self._compute_policy_values(temp_next_obs, temp_obs)
            else:
                current_pi_values1, current_pi_values2  = self._compute_policy_values(obs_pi=temp_actor_obs_features, obs_q=temp_critic_obs_features)
                next_pi_values1, next_pi_values2 = self._compute_policy_values(obs_pi=temp_actor_next_obs_features, obs_q=temp_critic_obs_features)
        
        if self.goal_conditioned:
            random_values1,random_values2 = self._compute_random_values(temp_obs, random_actions, goals=temp_goals)
        else:
            if len(obs.shape) == 2:
                random_values1,random_values2 = self._compute_random_values(temp_obs, random_actions)
            else:
                random_values1,random_values2 = self._compute_random_values(obs = temp_critic_obs_features, actions=random_actions)
        random_values1 = random_values1.view(obs.shape[0], num_repeat, 1)
        random_values2 = random_values2.view(obs.shape[0], num_repeat, 1)
        
        current_pi_values1 = current_pi_values1.reshape(obs.shape[0], num_repeat, 1)
        current_pi_values2 = current_pi_values2.reshape(obs.shape[0], num_repeat, 1)

        next_pi_values1 = next_pi_values1.reshape(obs.shape[0], num_repeat, 1)
        next_pi_values2 = next_pi_values2.reshape(obs.shape[0], num_repeat, 1)
        
        cat_q1 = torch.cat([random_values1, current_pi_values1, next_pi_values1], 1)
        cat_q2 = torch.cat([random_values2, current_pi_values2, next_pi_values2], 1)
        
        assert cat_q1.shape == (obs.shape[0], 3 * num_repeat, 1), f"cat_q1 instead has shape: {cat_q1.shape}"
        assert cat_q2.shape == (obs.shape[0], 3 * num_repeat, 1), f"cat_q2 instead has shape: {cat_q2.shape}"
        

        cql1_scaled_loss = ((torch.logsumexp(cat_q1 / self.temp, dim=1).mean() * self.cql_weight * self.temp) - qf1.mean()) * self.cql_weight
        cql2_scaled_loss = ((torch.logsumexp(cat_q2 / self.temp, dim=1).mean() * self.cql_weight * self.temp) - qf2.mean()) * self.cql_weight
        
        cql_alpha_loss = torch.FloatTensor([0.0])
        cql_alpha = torch.FloatTensor([0.0])
        if self.with_lagrange:
            cql_alpha = torch.clamp(self.cql_log_alpha.exp(), min=0.0, max=1000000.0).to(self.device)
            cql1_scaled_loss = cql_alpha * (cql1_scaled_loss - self.target_action_gap)
            cql2_scaled_loss = cql_alpha * (cql2_scaled_loss - self.target_action_gap)

            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss = (- cql1_scaled_loss - cql2_scaled_loss) * 0.5 
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optimizer.step()
        

        
        
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        with torch.no_grad():
            td_error = 0.5 * (torch.abs(qf2 - next_q_value) + torch.abs(qf1 - next_q_value)).mean().item()


        total_c1_loss = qf1_loss + cql1_scaled_loss
        total_c2_loss = qf2_loss + cql2_scaled_loss

        avg_q_values_1 = qf1.mean().item()
        avg_q_values_2 = qf2.mean().item()

        return total_c1_loss, total_c2_loss, td_error, qf1_loss, qf2_loss, cql1_scaled_loss, cql2_scaled_loss, avg_q_values_1, avg_q_values_2
    


    def _compute_policy_values(self, obs_pi, obs_q, goals=None):
        if self.goal_conditioned:
            actions_pred, log_pis,_ = self.actor.sample(obs_pi,goals)
            qs1,qs2 = self.critic(obs_q, actions_pred,goals)
        else:
            try:
                actions_pred, log_pis,_ = self.actor.sample_head(obs_pi)
                qs1,qs2 = self.critic.forward_critic_head(obs_q, actions_pred)
            except:
                actions_pred, log_pis,_ = self.actor.sample(obs_pi)
                qs1,qs2 = self.critic(obs_q, actions_pred)
        
        return qs1 - log_pis.detach(), qs2 - log_pis.detach()
    

    

    def _compute_random_values(self, obs, actions,goals=None):
        if self.goal_conditioned:
            random_values1,random_values2 = self.critic(obs, actions,goals=goals)
        else:

            try:
                random_values1,random_values2 = self.critic.forward_critic_head(obs, actions)
            except:
                random_values1,random_values2 = self.critic(obs, actions)
        
        random_log_probs = math.log(0.5 ** self.n_a)
        return random_values1 - random_log_probs, random_values2 - random_log_probs



    def updateActorAndAlpha(self):
        """
        Update actor and alpha
        :return: policy_loss, alpha_loss, alpha
        """
        policy_loss = self.calcActorLoss()
        log_pi = self.loss_calc_dict['log_pi']

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        return policy_loss, alpha_loss, alpha_tlogs

    
    
    def updateCritic(self):
        """
        Update critic
        :return: q1 loss, q2 loss, td error
        """
        qf1_loss, qf2_loss, td_error, ind_qf1_loss, ind_qf2_loss, cql1_scaled_loss, cql2_scaled_loss, avg_q_values_1, avg_q_values_2  = self.calcCriticLoss()
        qf_loss = qf1_loss + qf2_loss

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        return qf1_loss, qf2_loss, td_error, ind_qf1_loss, ind_qf2_loss, cql1_scaled_loss, cql2_scaled_loss, avg_q_values_1, avg_q_values_2

    
    
    def update(self, batch):
        """
        Perform a training step
        :param batch: the sampled minibatch
        :return: loss
        """
        self._loadBatchToDevice(batch)
        qf1_loss, qf2_loss, td_error, ind_qf1_loss, ind_qf2_loss, cql1_scaled_loss, cql2_scaled_loss, avg_q_values_1, avg_q_values_2 = self.updateCritic()
        policy_loss, alpha_loss, alpha_tlogs = self.updateActorAndAlpha()

        self.num_update += 1
        if self.num_update % self.target_update_interval == 0:
            self.targetSoftUpdate()

        self.loss_calc_dict = {}

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(), td_error, ind_qf1_loss.item(), ind_qf2_loss.item(), cql1_scaled_loss.item(), cql2_scaled_loss.item(), avg_q_values_1, avg_q_values_2
