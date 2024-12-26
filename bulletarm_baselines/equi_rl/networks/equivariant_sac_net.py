import torch

from e2cnn import gspaces
from e2cnn import nn

import torch.nn as torch_nn
from torch.distributions import Normal
from bulletarm_baselines.equi_rl.networks.sac_net import SACGaussianPolicyBase

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class EquivariantEncoder128(torch.nn.Module):
    """
    Equivariant Encoder. The input is a trivial representation with obs_channel channels.
    The output is a regular representation with n_out channels
    """
    def __init__(self, obs_channel=2, n_out=128, initialize=True, N=4):
        super().__init__()
        self.obs_channel = obs_channel
        self.c4_act = gspaces.Rot2dOnR2(N)
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.c4_act, obs_channel * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out*2 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out*2 * [self.c4_act.regular_repr]), inplace=True),

            nn.R2Conv(nn.FieldType(self.c4_act, n_out*2 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
            # 1x1
        )

    def forward(self, geo):
        return self.conv(geo)

def getEnc(obs_size, enc_id):
    assert obs_size in [128]
    assert enc_id in [1]
    return EquivariantEncoder128

class EquivariantEncoder128Dihedral(torch.nn.Module):
  def __init__(self, obs_channel=2, n_out=128, initialize=True, N=4):
    super().__init__()
    self.obs_channel = obs_channel
    self.c4_act = gspaces.FlipRot2dOnR2(N)
    self.conv = torch.nn.Sequential(
      # 128x128
      nn.R2Conv(nn.FieldType(self.c4_act, obs_channel * [self.c4_act.trivial_repr]),
                nn.FieldType(self.c4_act, n_out // 8 * [self.c4_act.regular_repr]),
                kernel_size=3, padding=1, initialize=initialize),
      nn.ReLU(nn.FieldType(self.c4_act, n_out // 8 * [self.c4_act.regular_repr]), inplace=True),
      nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out // 8 * [self.c4_act.regular_repr]), 2),
      # 64x64
      nn.R2Conv(nn.FieldType(self.c4_act, n_out // 8 * [self.c4_act.regular_repr]),
                nn.FieldType(self.c4_act, n_out // 4 * [self.c4_act.regular_repr]),
                kernel_size=3, padding=1, initialize=initialize),
      nn.ReLU(nn.FieldType(self.c4_act, n_out // 4 * [self.c4_act.regular_repr]), inplace=True),
      nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out // 4 * [self.c4_act.regular_repr]), 2),
      # 32x32
      nn.R2Conv(nn.FieldType(self.c4_act, n_out // 4 * [self.c4_act.regular_repr]),
                nn.FieldType(self.c4_act, n_out // 2 * [self.c4_act.regular_repr]),
                kernel_size=3, padding=1, initialize=initialize),
      nn.ReLU(nn.FieldType(self.c4_act, n_out // 2 * [self.c4_act.regular_repr]), inplace=True),
      nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out // 2 * [self.c4_act.regular_repr]), 2),
      # 16x16
      nn.R2Conv(nn.FieldType(self.c4_act, n_out // 2 * [self.c4_act.regular_repr]),
                nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                kernel_size=3, padding=1, initialize=initialize),
      nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
      nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), 2),
      # 8x8
      nn.R2Conv(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                nn.FieldType(self.c4_act, n_out * 2 * [self.c4_act.regular_repr]),
                kernel_size=3, padding=1, initialize=initialize),
      nn.ReLU(nn.FieldType(self.c4_act, n_out * 2 * [self.c4_act.regular_repr]), inplace=True),

      nn.R2Conv(nn.FieldType(self.c4_act, n_out * 2 * [self.c4_act.regular_repr]),
                nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                kernel_size=3, padding=0, initialize=initialize),
      nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
      nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), 2),
      # 3x3
      nn.R2Conv(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                kernel_size=3, padding=0, initialize=initialize),
      nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
      # 1x1
    )

  def forward(self, geo):
    # geo = nn.GeometricTensor(x, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
    return self.conv(geo)




class EquivariantSACCritic(torch.nn.Module):
    """
    Equivariant SAC's invariant critic
    """
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, enc_id=1):
        super().__init__()
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.c4_act = gspaces.Rot2dOnR2(N)
        enc = getEnc(obs_shape[1], enc_id)
        self.img_conv = enc(self.obs_channel, n_hidden, initialize, N)
        self.n_rho1 = 2 if N==2 else 1

        self.critic_1 = torch.nn.Sequential(
            # mixed representation including n_hidden regular representations (for the state),
            # (action_dim-2) trivial representations (for the invariant actions)
            # and 1 standard representation (for the equivariant actions)
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + (action_dim-2) * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

        self.critic_2 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + (action_dim-2) * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

    def forward(self, obs, act):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        conv_out = self.img_conv(obs_geo)
        dxy = act[:, 1:3]
        inv_act = torch.cat((act[:, 0:1], act[:, 3:]), dim=1)
        n_inv = inv_act.shape[1]
        cat = torch.cat((conv_out.tensor, inv_act.reshape(batch_size, n_inv, 1, 1), dxy.reshape(batch_size, 2, 1, 1)), dim=1)
        cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.regular_repr] + n_inv * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]))
        out1 = self.critic_1(cat_geo).tensor.reshape(batch_size, 1)
        out2 = self.critic_2(cat_geo).tensor.reshape(batch_size, 1)
        return out1, out2
    






class EquivariantSACCriticCQL(torch.nn.Module):
    """
    Equivariant SAC's invariant critic
    """
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, enc_id=1):
        super().__init__()
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.c4_act = gspaces.Rot2dOnR2(N)
        enc = getEnc(obs_shape[1], enc_id)
        self.img_conv = enc(self.obs_channel, n_hidden, initialize, N)
        self.n_rho1 = 2 if N==2 else 1

        self.critic_1 = torch.nn.Sequential(
            # mixed representation including n_hidden regular representations (for the state),
            # (action_dim-2) trivial representations (for the invariant actions)
            # and 1 standard representation (for the equivariant actions)
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + (action_dim-2) * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

        self.critic_2 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + (action_dim-2) * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

    def forward(self, obs, act):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        conv_out = self.img_conv(obs_geo)
        dxy = act[:, 1:3]
        inv_act = torch.cat((act[:, 0:1], act[:, 3:]), dim=1)
        n_inv = inv_act.shape[1]
        cat = torch.cat((conv_out.tensor, inv_act.reshape(batch_size, n_inv, 1, 1), dxy.reshape(batch_size, 2, 1, 1)), dim=1)
        cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.regular_repr] + n_inv * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]))
        out1 = self.critic_1(cat_geo).tensor.reshape(batch_size, 1)
        out2 = self.critic_2(cat_geo).tensor.reshape(batch_size, 1)
        return out1, out2
    

    def forward_features(self, obs):
        '''
        Return GeometricTensor of the features
        '''

        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        conv_out = self.img_conv(obs_geo)
        return conv_out.tensor
    

    def forward_critic_head(self, features, act):
        '''
        Return the critic output
        '''
        batch_size = features.shape[0]

        dxy = act[:, 1:3]
        inv_act = torch.cat((act[:, 0:1], act[:, 3:]), dim=1)
        n_inv = inv_act.shape[1]
        cat = torch.cat((features, inv_act.reshape(batch_size, n_inv, 1, 1), dxy.reshape(batch_size, 2, 1, 1)), dim=1)
        cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.regular_repr] + n_inv * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]))
        out1 = self.critic_1(cat_geo).tensor.reshape(batch_size, 1)
        out2 = self.critic_2(cat_geo).tensor.reshape(batch_size, 1)
        return out1, out2




    


class EquivariantSACCriticGoal(torch.nn.Module):
    """
    Equivariant SAC's invariant critic
    """
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4,n_tasks=2,enc_id=1):
        super().__init__()
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.c4_act = gspaces.Rot2dOnR2(N)
        enc = getEnc(obs_shape[1], enc_id)
        self.img_conv = enc(self.obs_channel, n_hidden, initialize, N)
        self.n_rho1 = 2 if N==2 else 1
        self.n_tasks = n_tasks

        self.critic_1 = torch.nn.Sequential(
            # mixed representation including n_hidden regular representations (for the state),
            # (action_dim-2) trivial representations (for the invariant actions)
            # and 1 standard representation (for the equivariant actions)
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + (action_dim-2) * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)] + n_tasks*[self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

        self.critic_2 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + (action_dim-2) * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)] + n_tasks*[self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

    def forward(self, obs, act,goals):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        conv_out = self.img_conv(obs_geo)
        dxy = act[:, 1:3]
        inv_act = torch.cat((act[:, 0:1], act[:, 3:]), dim=1)
        n_inv = inv_act.shape[1]
        cat = torch.cat((conv_out.tensor, inv_act.reshape(batch_size, n_inv, 1, 1), dxy.reshape(batch_size, 2, 1, 1), goals.reshape(batch_size,self.n_tasks, 1, 1)), dim=1)
        cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.regular_repr] + n_inv * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)] + self.n_tasks*[self.c4_act.trivial_repr]))
        out1 = self.critic_1(cat_geo).tensor.reshape(batch_size, 1)
        out2 = self.critic_2(cat_geo).tensor.reshape(batch_size, 1)
        return out1, out2



class EquivariantSACCriticDihedral(torch.nn.Module):
  def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4):
    super().__init__()
    self.obs_channel = obs_shape[0]
    self.n_hidden = n_hidden
    self.c4_act = gspaces.FlipRot2dOnR2(N)
    enc = EquivariantEncoder128Dihedral
    self.img_conv = enc(self.obs_channel, n_hidden, initialize, N)
    self.n_rho1 = 2 if N == 2 else 1
    self.critic_1 = torch.nn.Sequential(
      nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + (action_dim - 2) * [
        self.c4_act.trivial_repr] + self.n_rho1 * [self.c4_act.irrep(1, 1)]),
                nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                kernel_size=1, padding=0, initialize=initialize),
      nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
      nn.GroupPooling(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr])),
      nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                kernel_size=1, padding=0, initialize=initialize),
    )

    self.critic_2 = torch.nn.Sequential(
      nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + (action_dim - 2) * [
        self.c4_act.trivial_repr] + self.n_rho1 * [self.c4_act.irrep(1, 1)]),
                nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                kernel_size=1, padding=0, initialize=initialize),
      nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
      nn.GroupPooling(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr])),
      nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                kernel_size=1, padding=0, initialize=initialize),
    )

  def forward(self, obs, act):
    batch_size = obs.shape[0]
    obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel * [self.c4_act.trivial_repr]))
    conv_out = self.img_conv(obs_geo)
    dxy = act[:, 1:3]
    inv_act = torch.cat((act[:, 0:1], act[:, 3:]), dim=1)
    n_inv = inv_act.shape[1]
    # dxy_geo = nn.GeometricTensor(dxy.reshape(batch_size, 2, 1, 1), nn.FieldType(self.c4_act, 1*[self.c4_act.irrep(1)]))
    # inv_act_geo = nn.GeometricTensor(inv_act.reshape(batch_size, n_inv, 1, 1), nn.FieldType(self.c4_act, n_inv*[self.c4_act.trivial_repr]))
    cat = torch.cat((conv_out.tensor, inv_act.reshape(batch_size, n_inv, 1, 1), dxy.reshape(batch_size, 2, 1, 1)),
                    dim=1)
    cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.regular_repr] + n_inv * [
      self.c4_act.trivial_repr] + self.n_rho1 * [self.c4_act.irrep(1, 1)]))
    out1 = self.critic_1(cat_geo).tensor.reshape(batch_size, 1)
    out2 = self.critic_2(cat_geo).tensor.reshape(batch_size, 1)
    return out1, out2



class EquivariantSACValueCritic(torch.nn.Module):
    """
    Equivariant SAC's invariant critic
    """
    def __init__(self, obs_shape=(2, 128, 128), n_hidden=128, initialize=True, N=4, enc_id=1):
        super().__init__()
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.c4_act = gspaces.Rot2dOnR2(N)
        enc = getEnc(obs_shape[1], enc_id)
        self.img_conv = enc(self.obs_channel, n_hidden, initialize, N)
        self.n_rho1 = 2 if N==2 else 1

        self.v_1 = torch.nn.Sequential(
            # mixed representation including n_hidden regular representations (for the state),
            # (action_dim-2) trivial representations (for the invariant actions)
            # and 1 standard representation (for the equivariant actions)
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),

            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr])),   # Leaving the GroupPooling here for now
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        conv_out = self.img_conv(obs_geo)
        # cat = torch.cat((conv_out.tensor, inv_act.reshape(batch_size, n_inv, 1, 1), dxy.reshape(batch_size, 2, 1, 1)), dim=1)
        # cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.regular_repr] + n_inv * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]))
        v = self.v_1(conv_out).tensor.reshape(batch_size, 1)
        # out2 = self.critic_2(cat_geo).tensor.reshape(batch_size, 1)
        
        
        return v
    

class EquivariantSACValueCriticGoal(torch.nn.Module):
    """
    Equivariant SAC's invariant critic
    """
    def __init__(self, obs_shape=(2, 128, 128), n_hidden=128, initialize=True, N=4, n_tasks=2, enc_id=1):
        super().__init__()
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.c4_act = gspaces.Rot2dOnR2(N)
        enc = getEnc(obs_shape[1], enc_id)
        self.img_conv = enc(self.obs_channel, n_hidden, initialize, N)
        self.n_rho1 = 2 if N==2 else 1
        self.n_tasks = n_tasks

        self.v_1 = torch.nn.Sequential(
            # mixed representation including n_hidden regular representations (for the state),
            # (action_dim-2) trivial representations (for the invariant actions)
            # and 1 standard representation (for the equivariant actions)
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + self.n_tasks*[self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),

            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr])),   # Leaving the GroupPooling here for now
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

    def forward(self, obs, goals):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        conv_out = self.img_conv(obs_geo)
        # cat = torch.cat((conv_out.tensor, inv_act.reshape(batch_size, n_inv, 1, 1), dxy.reshape(batch_size, 2, 1, 1)), dim=1)

        cat = torch.cat((conv_out.tensor, goals.reshape(batch_size, self.n_tasks, 1, 1)), dim=1)
        cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.regular_repr] + self.n_tasks * [self.c4_act.trivial_repr]))
        v = self.v_1(cat_geo).tensor.reshape(batch_size, 1)

        return v
    
    
  


class EquivariantSACValueCritic2(torch.nn.Module):
    """
    Equivariant SAC's invariant critic
    """
    def __init__(self, obs_shape=(2, 128, 128), n_hidden=128, initialize=True, N=4, enc_id=1):
        super().__init__()
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.c4_act = gspaces.Rot2dOnR2(N)
        enc = getEnc(obs_shape[1], enc_id)
        self.img_conv = enc(self.obs_channel, n_hidden, initialize, N)
        self.n_rho1 = 2 if N==2 else 1

        self.v_1 = torch.nn.Sequential(
            # mixed representation including n_hidden regular representations (for the state),
            # (action_dim-2) trivial representations (for the invariant actions)
            # and 1 standard representation (for the equivariant actions)
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),

            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),\
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        conv_out = self.img_conv(obs_geo)
        # cat = torch.cat((conv_out.tensor, inv_act.reshape(batch_size, n_inv, 1, 1), dxy.reshape(batch_size, 2, 1, 1)), dim=1)
        # cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.regular_repr] + n_inv * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]))
        v = self.v_1(conv_out).tensor.reshape(batch_size, 1)
        # out2 = self.critic_2(cat_geo).tensor.reshape(batch_size, 1)
        
        
        return v
    



class EquivariantSACValueCritic2Goal(torch.nn.Module):
    """
    Equivariant SAC's invariant critic
    """
    def __init__(self, obs_shape=(2, 128, 128), n_hidden=128, initialize=True, N=4, n_tasks=2, enc_id=1):
        super().__init__()
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.c4_act = gspaces.Rot2dOnR2(N)
        enc = getEnc(obs_shape[1], enc_id)
        self.img_conv = enc(self.obs_channel, n_hidden, initialize, N)
        self.n_rho1 = 2 if N==2 else 1
        self.n_tasks = n_tasks

        self.v_1 = torch.nn.Sequential(
            # mixed representation including n_hidden regular representations (for the state),
            # (action_dim-2) trivial representations (for the invariant actions)
            # and 1 standard representation (for the equivariant actions)
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + self.n_tasks*[self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),

            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

    def forward(self, obs, goals):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        conv_out = self.img_conv(obs_geo)
        # cat = torch.cat((conv_out.tensor, inv_act.reshape(batch_size, n_inv, 1, 1), dxy.reshape(batch_size, 2, 1, 1)), dim=1)

        cat = torch.cat((conv_out.tensor, goals.reshape(batch_size, self.n_tasks, 1, 1)), dim=1)
        cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.regular_repr] + self.n_tasks * [self.c4_act.trivial_repr]))
        v = self.v_1(cat_geo).tensor.reshape(batch_size, 1)

        return v
    





class EquivariantSACActor(SACGaussianPolicyBase):
    """
    Equivariant SAC's equivariant actor
    """
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, enc_id=1):
        super().__init__()
        assert obs_shape[1] in [128, 64]
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.c4_act = gspaces.Rot2dOnR2(N)
        enc = getEnc(obs_shape[1], enc_id)
        self.n_rho1 = 2 if N==2 else 1
        self.conv = torch.nn.Sequential(
            enc(self.obs_channel, n_hidden, initialize, N),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      # mixed representation including action_dim trivial representations (for the std of all actions),
                      # (action_dim-2) trivial representations (for the mu of invariant actions),
                      # and 1 standard representation (for the mu of equivariant actions)
                      nn.FieldType(self.c4_act, self.n_rho1 * [self.c4_act.irrep(1)] + (action_dim*2-2) * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        conv_out = self.conv(obs_geo).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        inv_act = conv_out[:, 2:self.action_dim]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
        log_std = conv_out[:, self.action_dim:]
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std
    





class EquivariantSACActorCQL(SACGaussianPolicyBase):
    """
    Equivariant SAC's equivariant actor
    """
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, enc_id=1):
        super().__init__()
        assert obs_shape[1] in [128, 64]
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.c4_act = gspaces.Rot2dOnR2(N)
        
        enc = getEnc(obs_shape[1], enc_id)
        self.img_conv = enc(self.obs_channel, n_hidden, initialize, N)
        self.n_rho1 = 2 if N==2 else 1

        self.n_hidden = n_hidden

        self.conv = torch.nn.Sequential(
            # enc(self.obs_channel, n_hidden, initialize, N),
            
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      # mixed representation including action_dim trivial representations (for the std of all actions),
                      # (action_dim-2) trivial representations (for the mu of invariant actions),
                      # and 1 standard representation (for the mu of equivariant actions)
                      nn.FieldType(self.c4_act, self.n_rho1 * [self.c4_act.irrep(1)] + (action_dim*2-2) * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

        

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        conv_out = self.img_conv(obs_geo)

        conv_out = self.conv(conv_out).tensor.reshape(batch_size, -1)
        
        
        dxy = conv_out[:, 0:2]
        inv_act = conv_out[:, 2:self.action_dim]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
        log_std = conv_out[:, self.action_dim:]
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std
    

    def forward_features(self, obs):
        '''
        Return GeometricTensor of the features
        '''

        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        conv_out = self.img_conv(obs_geo)
        return conv_out.tensor
    

    def forward_actor_head(self, features):
        '''
        Return the actor output
        '''
        batch_size = features.shape[0]

        features_geo = nn.GeometricTensor(features, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.regular_repr]))
        conv_out = self.conv(features_geo).tensor.reshape(batch_size, -1)
        
        dxy = conv_out[:, 0:2]
        inv_act = conv_out[:, 2:self.action_dim]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
        log_std = conv_out[:, self.action_dim:]
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std
    

    def sample_head(self, x):
        mean, log_std = self.forward_actor_head(x)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean




  



class EquivariantSACActorGoal(SACGaussianPolicyBase):
    """
    Equivariant SAC's equivariant actor
    """
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, n_tasks=2, enc_id=1):
        super().__init__()
        assert obs_shape[1] in [128, 64]
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.c4_act = gspaces.Rot2dOnR2(N)
        self.n_rho1 = 2 if N==2 else 1
        self.n_hidden = n_hidden

        enc = getEnc(obs_shape[1], enc_id)
        self.img_conv = enc(self.obs_channel, n_hidden, initialize, N)
        self.n_tasks = n_tasks

        self.conv = torch.nn.Sequential(
                      nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + self.n_tasks*[self.c4_act.trivial_repr]),
                      
                      # mixed representation including action_dim trivial representations (for the std of all actions),
                      # (action_dim-2) trivial representations (for the mu of invariant actions),
                      # and 1 standard representation (for the mu of equivariant actions)
                      
                      nn.FieldType(self.c4_act, self.n_rho1 * [self.c4_act.irrep(1)] + (action_dim*2-2) * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs, goals):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        conv_out = self.img_conv(obs_geo)

        concat_tensor = torch.cat((conv_out.tensor, goals.reshape(batch_size, self.n_tasks, 1, 1)), dim=1)
        concat_geo = nn.GeometricTensor(concat_tensor, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.regular_repr]+ (self.n_tasks)*[self.c4_act.trivial_repr]))

        conv_out = self.conv(concat_geo).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        inv_act = conv_out[:, 2:self.action_dim]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
        log_std = conv_out[:, self.action_dim:]

        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std
    

    def get_logprobs(self, x,actions, goals):
        mean, log_std = self.forward(x,goals)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t

        log_probs = normal.log_prob(actions)

        # Enforcing Action Bound
        log_probs -= torch.log((1 - y_t.pow(2)) + epsilon)
        log_probs = log_probs.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        return log_probs, mean, std
    

    def sample(self, x,goals):
        mean, log_std = self.forward(x,goals)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean
    




class EquivariantIQLActor(SACGaussianPolicyBase):
    """
    Equivariant SAC's equivariant actor
    """
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, enc_id=1):
        super().__init__()
        assert obs_shape[1] in [128, 64]
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.c4_act = gspaces.Rot2dOnR2(N)
        enc = getEnc(obs_shape[1], enc_id)
        self.n_rho1 = 2 if N==2 else 1
        self.conv = torch.nn.Sequential(
            enc(self.obs_channel, n_hidden, initialize, N),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      # mixed representation including action_dim trivial representations (for the std of all actions),
                      # (action_dim-2) trivial representations (for the mu of invariant actions),
                      # and 1 standard representation (for the mu of equivariant actions)
                      nn.FieldType(self.c4_act, self.n_rho1 * [self.c4_act.irrep(1)] + (action_dim-2) * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )
        # self.log_std = torch_nn.Parameter(torch.zeros(action_dim, requires_grad=True))
        self.log_std = torch_nn.Parameter(-1 * torch.ones(action_dim, requires_grad=True))


    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        conv_out = self.conv(obs_geo).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        inv_act = conv_out[:, 2:self.action_dim]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)

        
        log_std_expanded = self.log_std.expand(batch_size, -1)
        log_std = torch.clamp(log_std_expanded, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std




class EquivariantIQLActorGoal(SACGaussianPolicyBase):
    """
    Equivariant SAC's equivariant actor
    """
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, n_tasks=2, enc_id=1):
        super().__init__()
        assert obs_shape[1] in [128, 64]
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.c4_act = gspaces.Rot2dOnR2(N)
        self.n_rho1 = 2 if N==2 else 1
        self.n_hidden = n_hidden

        enc = getEnc(obs_shape[1], enc_id)
        self.img_conv = enc(self.obs_channel, n_hidden, initialize, N)
        self.n_tasks = n_tasks

        self.conv = torch.nn.Sequential(
                      nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + self.n_tasks*[self.c4_act.trivial_repr]),
                      
                      # mixed representation including action_dim trivial representations (for the std of all actions),
                      # (action_dim-2) trivial representations (for the mu of invariant actions),
                      # and 1 standard representation (for the mu of equivariant actions)
                      
                      nn.FieldType(self.c4_act, self.n_rho1 * [self.c4_act.irrep(1)] + (action_dim*2-2) * [self.c4_act.trivial_repr] + n_tasks*[self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )
        self.log_std = torch_nn.Parameter(-1 * torch.ones(action_dim, requires_grad=True))


    def forward(self, obs, goals):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        conv_out = self.img_conv(obs_geo)

        concat_tensor = torch.cat((conv_out.tensor, goals.reshape(batch_size, self.n_tasks, 1, 1)), dim=1)
        concat_geo = nn.GeometricTensor(concat_tensor, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.regular_repr]+ (self.n_tasks)*[self.c4_act.trivial_repr]))

        conv_out = self.conv(concat_geo).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        inv_act = conv_out[:, 2:self.action_dim]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
        
        log_std_expanded = self.log_std.expand(batch_size, -1)
        log_std = torch.clamp(log_std_expanded, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std
    

    def get_logprobs(self, x,actions, goals):
        mean, log_std = self.forward(x,goals)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t

        log_probs = normal.log_prob(actions)

        # Enforcing Action Bound
        log_probs -= torch.log((1 - y_t.pow(2)) + epsilon)
        log_probs = log_probs.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        return log_probs, mean, std
    

    def sample(self, x,goals):
        mean, log_std = self.forward(x,goals)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean






class EquivariantSACActorDihedral(SACGaussianPolicyBase):
  
  def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4):
    super().__init__()
    assert obs_shape[1] in [128, 64]
    self.obs_channel = obs_shape[0]
    self.action_dim = action_dim
    self.c4_act = gspaces.FlipRot2dOnR2(N)
    self.n_rho1 = 2 if N == 2 else 1
    enc = EquivariantEncoder128Dihedral
    self.conv = torch.nn.Sequential(
      enc(self.obs_channel, n_hidden, initialize, N),
      nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                nn.FieldType(self.c4_act, self.n_rho1 * [self.c4_act.irrep(1, 1)] + (action_dim * 2 - 2) * [
                  self.c4_act.trivial_repr]),
                kernel_size=1, padding=0, initialize=initialize)
    )

  def forward(self, obs):
    batch_size = obs.shape[0]
    obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel * [self.c4_act.trivial_repr]))
    conv_out = self.conv(obs_geo).tensor.reshape(batch_size, -1)
    dxy = conv_out[:, 0:2]
    inv_act = conv_out[:, 2:self.action_dim]
    mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
    log_std = conv_out[:, self.action_dim:]
    log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
    return mean, log_std



class EquivariantSACCritic_2_Objects(torch.nn.Module):
    """
    Equivariant SAC's invariant critic
    """
    def __init__(self, obs_shape=(16,), action_dim=5, n_hidden=128, initialize=True, N=4, enc_id=1):
        super().__init__()
        self.input_dims = obs_shape[0]
        self.n_hidden = n_hidden
        self.c4_act = gspaces.Rot2dOnR2(N)
        self.n_rho1 = 2 if N==2 else 1

        self.critic_1 = torch.nn.Sequential(

            # mixed representation including n_hidden regular representations (for the state),
            # (action_dim-2) trivial representations (for the invariant actions)
            # and 1 standard representation (for the equivariant actions)
            nn.R2Conv(nn.FieldType(self.c4_act, 4 * [self.c4_act.trivial_repr] + 6 * [self.c4_act.irrep(1)] + (action_dim-2) * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

        self.critic_2 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, 4 * [self.c4_act.trivial_repr] + 6 * [self.c4_act.irrep(1)] + (action_dim-2) * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )
      
    def getInFieldType(self):
        return nn.FieldType(
            self.c4_act,
            4 * [self.c4_act.trivial_repr] # 4
            + 6 * [self.group.irrep(1)] # 12
        )

    def forward(self, obs, act):
        batch_size = obs.shape[0]

        obs_gripper_state = obs[:,0].reshape(batch_size, 1, 1, 1)

        obs_gripper_xy = obs[:,1:3].reshape(batch_size,2, 1, 1)
        obs_gripper_z = obs[:,3].reshape(batch_size, 1, 1, 1)
        obs_gripper_c_s_rz = obs[:,4:6].reshape(batch_size,2, 1, 1)

        obs_obj1_xy = obs[:,6:8].reshape(batch_size,2, 1, 1)
        obs_obj1_z = obs[:,8].reshape(batch_size, 1, 1, 1)
        obs_obj1_c_s_rz = obs[:,9:11].reshape(batch_size,2, 1, 1)

        obs_obj2_xy = obs[:,11:13].reshape(batch_size,2, 1, 1)
        obs_obj2_z = obs[:,13].reshape(batch_size, 1, 1, 1)
        obs_obj2_c_s_rz = obs[:,14:16].reshape(batch_size,2, 1, 1)

        final_obs = torch.cat((obs_gripper_state, obs_gripper_z, obs_obj1_z, obs_obj2_z,
                              obs_gripper_xy, obs_obj1_xy, obs_obj2_xy, obs_gripper_c_s_rz, obs_obj1_c_s_rz, obs_obj2_c_s_rz), dim=1) 
                               

        dxy = act[:, 1:3]
        inv_act = torch.cat((act[:, 0:1], act[:, 3:]), dim=1)
        n_inv = inv_act.shape[1]
        cat = torch.cat((final_obs, inv_act.reshape(batch_size, n_inv, 1, 1),dxy.reshape(batch_size, 2, 1, 1)), dim=1)
        cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.c4_act,4 * [self.c4_act.trivial_repr] + 6 * [self.c4_act.irrep(1)] + n_inv * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]))
        out1 = self.critic_1(cat_geo).tensor.reshape(batch_size, 1)
        out2 = self.critic_2(cat_geo).tensor.reshape(batch_size, 1)
        return out1, out2
    



class EquivariantSACCValueCritic_2_Objects(torch.nn.Module):
    """
    Equivariant IQL's invariant value fn
    """
    def __init__(self, obs_shape=(16,), action_dim=5, n_hidden=128, initialize=True, N=4, enc_id=1):
        super().__init__()
        self.input_dims = obs_shape[0]
        self.n_hidden = n_hidden
        self.c4_act = gspaces.Rot2dOnR2(N)
        self.n_rho1 = 2 if N==2 else 1

        self.value_fn = torch.nn.Sequential(
            # mixed representation including n_hidden regular representations (for the state),
            # (action_dim-2) trivial representations (for the invariant actions)
            # and 1 standard representation (for the equivariant actions)
            nn.R2Conv(nn.FieldType(self.c4_act, 4 * [self.c4_act.trivial_repr] + 6 * [self.c4_act.irrep(1)]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )
      
    def getInFieldType(self):
        return nn.FieldType(
            self.c4_act,
            4 * [self.c4_act.trivial_repr] # 4
            + 6 * [self.group.irrep(1)] # 12
        )

    def forward(self, obs):
        batch_size = obs.shape[0]

        obs_gripper_state = obs[:,0].reshape(batch_size, 1, 1, 1)

        obs_gripper_xy = obs[:,1:3].reshape(batch_size,2, 1, 1)
        obs_gripper_z = obs[:,3].reshape(batch_size, 1, 1, 1)
        obs_gripper_c_s_rz = obs[:,4:6].reshape(batch_size,2, 1, 1)

        obs_obj1_xy = obs[:,6:8].reshape(batch_size,2, 1, 1)
        obs_obj1_z = obs[:,8].reshape(batch_size, 1, 1, 1)
        obs_obj1_c_s_rz = obs[:,9:11].reshape(batch_size,2, 1, 1)

        obs_obj2_xy = obs[:,11:13].reshape(batch_size,2, 1, 1)
        obs_obj2_z = obs[:,13].reshape(batch_size, 1, 1, 1)
        obs_obj2_c_s_rz = obs[:,14:16].reshape(batch_size,2, 1, 1)

        
        final_obs = torch.cat((obs_gripper_state, obs_gripper_z, obs_obj1_z, obs_obj2_z,
                              obs_gripper_xy, obs_obj1_xy, obs_obj2_xy, obs_gripper_c_s_rz, obs_obj1_c_s_rz, obs_obj2_c_s_rz), dim=1) 
        
        cat_geo = nn.GeometricTensor(final_obs, nn.FieldType(self.c4_act,4 * [self.c4_act.trivial_repr] + 6 * [self.c4_act.irrep(1)]))
        out = self.value_fn(cat_geo).tensor.reshape(batch_size, 1)
        return out
                               
        
    




class EquivariantSACActor_2_Objects(SACGaussianPolicyBase):
    """
    Equivariant SAC's equivariant actor
    """
    def __init__(self, obs_shape=(16,), action_dim=5, n_hidden=128, initialize=True, N=4, enc_id=1):
        super().__init__()
        self.input_dims = obs_shape[0]
        self.action_dim = action_dim
        self.c4_act = gspaces.Rot2dOnR2(N)
        self.n_rho1 = 2 if N==2 else 1
        
        
        self.conv = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, 4 * [self.c4_act.trivial_repr] + 6 * [self.c4_act.irrep(1)]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            # mixed representation including action_dim trivial representations (for the std of all actions),
            # (action_dim-2) trivial representations (for the mu of invariant actions),
            # and 1 standard representation (for the mu of equivariant actions)
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, self.n_rho1 * [self.c4_act.irrep(1)] + (action_dim*2-2) * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):

        batch_size = obs.shape[0]

        obs_gripper_state = obs[:,0].reshape(batch_size, 1,1,1)

        obs_gripper_xy = obs[:,1:3].reshape(batch_size, 2, 1, 1)
        obs_gripper_z = obs[:,3].reshape(batch_size, 1, 1, 1)
        obs_gripper_c_s_rz = obs[:,4:6].reshape(batch_size, 2, 1, 1)

        obs_obj1_xy = obs[:,6:8].reshape(batch_size, 2, 1, 1)
        obs_obj1_z = obs[:,8].reshape(batch_size, 1, 1, 1)
        obs_obj1_c_s_rz = obs[:,9:11].reshape(batch_size, 2, 1, 1)

        obs_obj2_xy = obs[:,11:13].reshape(batch_size, 2, 1, 1)
        obs_obj2_z = obs[:,13].reshape(batch_size, 1, 1, 1)
        obs_obj2_c_s_rz = obs[:,14:16].reshape(batch_size, 2, 1, 1)


        final_obs = torch.cat((obs_gripper_state, obs_gripper_z, obs_obj1_z, obs_obj2_z,obs_gripper_xy, obs_obj1_xy, obs_obj2_xy, obs_gripper_c_s_rz, obs_obj1_c_s_rz, obs_obj2_c_s_rz), dim=1)
        obs_geo = nn.GeometricTensor(final_obs, nn.FieldType(self.c4_act,4 * [self.c4_act.trivial_repr] + 6 * [self.c4_act.irrep(1)]))
        
        conv_out = self.conv(obs_geo).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        inv_act = conv_out[:, 2:self.action_dim]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
        log_std = conv_out[:, self.action_dim:]
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std
    





class EquivariantSACCritic_1_Object(torch.nn.Module):
    """
    Equivariant SAC's invariant critic
    """
    def __init__(self, obs_shape=(16,), action_dim=5, n_hidden=128, initialize=True, N=4, enc_id=1):
        super().__init__()
        self.input_dims = obs_shape[0]
        self.n_hidden = n_hidden
        self.c4_act = gspaces.Rot2dOnR2(N)
        self.n_rho1 = 2 if N==2 else 1

        self.critic_1 = torch.nn.Sequential(

            # mixed representation including n_hidden regular representations (for the state),
            # (action_dim-2) trivial representations (for the invariant actions)
            # and 1 standard representation (for the equivariant actions)
            nn.R2Conv(nn.FieldType(self.c4_act, 3 * [self.c4_act.trivial_repr] + 4 * [self.c4_act.irrep(1)] + (action_dim-2) * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

        self.critic_2 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, 3 * [self.c4_act.trivial_repr] + 4 * [self.c4_act.irrep(1)] + (action_dim-2) * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )
      
    def getInFieldType(self):
        return nn.FieldType(
            self.c4_act,
            4 * [self.c4_act.trivial_repr] # 4
            + 6 * [self.group.irrep(1)] # 12
        )

    def forward(self, obs, act):
        batch_size = obs.shape[0]

        obs_gripper_state = obs[:,0].reshape(batch_size, 1, 1, 1)

        obs_gripper_xy = obs[:,1:3].reshape(batch_size,2, 1, 1)
        obs_gripper_z = obs[:,3].reshape(batch_size, 1, 1, 1)
        obs_gripper_c_s_rz = obs[:,4:6].reshape(batch_size,2, 1, 1)

        obs_obj1_xy = obs[:,6:8].reshape(batch_size,2, 1, 1)
        obs_obj1_z = obs[:,8].reshape(batch_size, 1, 1, 1)
        obs_obj1_c_s_rz = obs[:,9:11].reshape(batch_size,2, 1, 1)

        final_obs = torch.cat((obs_gripper_state, obs_gripper_z, obs_obj1_z,
                              obs_gripper_xy, obs_obj1_xy, obs_gripper_c_s_rz, obs_obj1_c_s_rz), dim=1) 
                               

        dxy = act[:, 1:3]
        inv_act = torch.cat((act[:, 0:1], act[:, 3:]), dim=1)
        n_inv = inv_act.shape[1]
        cat = torch.cat((final_obs, inv_act.reshape(batch_size, n_inv, 1, 1),dxy.reshape(batch_size, 2, 1, 1)), dim=1)
        cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.c4_act,3 * [self.c4_act.trivial_repr] + 4 * [self.c4_act.irrep(1)] + n_inv * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]))
        out1 = self.critic_1(cat_geo).tensor.reshape(batch_size, 1)
        out2 = self.critic_2(cat_geo).tensor.reshape(batch_size, 1)
        return out1, out2
    



class EquivariantSACActor_1_Object(SACGaussianPolicyBase):
    """
    Equivariant SAC's equivariant actor
    """
    def __init__(self, obs_shape=(16,), action_dim=5, n_hidden=128, initialize=True, N=4, enc_id=1):
        super().__init__()
        self.input_dims = obs_shape[0]
        self.action_dim = action_dim
        self.c4_act = gspaces.Rot2dOnR2(N)
        self.n_rho1 = 2 if N==2 else 1
        
        
        self.conv = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, 3 * [self.c4_act.trivial_repr] + 4 * [self.c4_act.irrep(1)]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),

            # mixed representation including action_dim trivial representations (for the std of all actions),
            # (action_dim-2) trivial representations (for the mu of invariant actions),
            # and 1 standard representation (for the mu of equivariant actions)
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, self.n_rho1 * [self.c4_act.irrep(1)] + (action_dim*2-2) * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]

        obs_gripper_state = obs[:,0].reshape(batch_size, 1,1,1)

        obs_gripper_xy = obs[:,1:3].reshape(batch_size, 2, 1, 1)
        obs_gripper_z = obs[:,3].reshape(batch_size, 1, 1, 1)
        obs_gripper_c_s_rz = obs[:,4:6].reshape(batch_size, 2, 1, 1)

        obs_obj1_xy = obs[:,6:8].reshape(batch_size, 2, 1, 1)
        obs_obj1_z = obs[:,8].reshape(batch_size, 1, 1, 1)
        obs_obj1_c_s_rz = obs[:,9:11].reshape(batch_size, 2, 1, 1)

        
        final_obs = torch.cat((obs_gripper_state, obs_gripper_z, obs_obj1_z,obs_gripper_xy, obs_obj1_xy, obs_gripper_c_s_rz, obs_obj1_c_s_rz), dim=1)
        obs_geo = nn.GeometricTensor(final_obs, nn.FieldType(self.c4_act,3 * [self.c4_act.trivial_repr] + 4 * [self.c4_act.irrep(1)]))
        
        conv_out = self.conv(obs_geo).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        inv_act = conv_out[:, 2:self.action_dim]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
        log_std = conv_out[:, self.action_dim:]
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std
    




if __name__ == "__main__":
    # test
    obs_shape = (2, 128, 128)
    action_dim = 5
    n_hidden = 128
    N = 8
    enc_id = 1
    critic = EquivariantSACCritic(obs_shape, action_dim, n_hidden, True, N, enc_id)


    obs = torch.zeros(1, *obs_shape)
    obs[:,:,10:20,10:20] = 1
    
    act = torch.tensor([1,1,1,1,1])
    act = act.reshape(1, -1).float()
    print("Observation Shape:", obs.shape)
    print("Action Shape:", act.shape)
    print()


    q1, q2 = critic(obs, act)
    print(q1.item(), q2.item())



    g_obs = torch.zeros(1, *obs_shape)
    g_obs[:,:,-20:-10,-20:-10] = 1

    g_act = torch.tensor([1,-1,-1,1,1])
    g_act = g_act.reshape(1, -1).float()
    print("Observation Shape:", g_obs.shape)
    print("Action Shape:", g_act.shape)

    g_q1, g_q2 = critic(g_obs, g_act)
    print(g_q1.item(), g_q2.item())