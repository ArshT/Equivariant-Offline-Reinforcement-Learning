from bulletarm_baselines.equi_rl.agents.sac import SAC
import numpy as np
import torch.nn.functional as F

class BCfD(SAC):
    """
    BCfD agent class
    """
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/16, n_a=5, tau=0.001,
                 alpha=0.01, policy_type='gaussian', target_update_interval=1, automatic_entropy_tuning=False,
                 obs_type='pixel', demon_w=0.1, demon_l='pi'):
        super().__init__(lr, gamma, device, dx, dy, dz, dr, n_a, tau, alpha, policy_type, target_update_interval,
                         automatic_entropy_tuning, obs_type)
        self.demon_w = demon_w
        assert demon_l in ['mean', 'pi']
        self.demon_l = demon_l

    def calcActorLoss(self):
        policy_loss = super().calcActorLoss()
        pi = self.loss_calc_dict['pi']
        mean = self.loss_calc_dict['mean']
        action = self.loss_calc_dict['action_idx']
        is_experts = self.loss_calc_dict['is_experts']
        # add expert loss
        if is_experts.sum():
            if self.demon_l == 'pi':
                demon_loss = F.mse_loss(pi[is_experts], action[is_experts])
                # policy_loss += self.demon_w * demon_loss
                policy_loss = demon_loss
            else:
                demon_loss = F.mse_loss(mean[is_experts], action[is_experts])
                # policy_loss += self.demon_w * demon_loss
                policy_loss = demon_loss
        return policy_loss
    


    def update(self, batch):
        """
        Perform a training step
        :param batch: the sampled minibatch
        :return: loss
        """
        self._loadBatchToDevice(batch)
        qf1_loss, qf2_loss, td_error = self.updateCritic()
        policy_loss, alpha_loss, alpha_tlogs = self.updateActorAndAlpha()

        self.num_update += 1
        if self.num_update % self.target_update_interval == 0:
            self.targetSoftUpdate()

        self.loss_calc_dict = {}

        return policy_loss.item()

