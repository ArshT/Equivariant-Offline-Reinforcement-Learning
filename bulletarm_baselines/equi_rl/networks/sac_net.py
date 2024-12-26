import torch
import torch.nn as nn
from torch.distributions import Normal
from bulletarm_baselines.equi_rl.utils import torch_utils
import torch.nn.functional as F

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class SACGaussianPolicyBase(nn.Module):
    def __init__(self):
        super().__init__()

    def sample(self, x):
        mean, log_std = self.forward(x)
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
    
    def get_logprobs(self, x,actions):
        mean, log_std = self.forward(x)
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

# similar amount of parameters
class SACEncoder(nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), out_dim=1024):
        super().__init__()
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.Conv2d(obs_shape[0], 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 64x64
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            # 6x6
            nn.MaxPool2d(2),
            # 3x3
            nn.Conv2d(256, out_dim, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.conv(x)

# similar amount of parameters
class SACCritic(nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5):
        super().__init__()
        print("Non-Equi Q Network")
        self.state_conv_1 = SACEncoder(obs_shape, 128)

        # Q1
        self.critic_fc_1 = torch.nn.Sequential(
            torch.nn.Linear(128+action_dim, 128),
            nn.ReLU(inplace=True),
            torch.nn.Linear(128, 1)
        )

        # Q2
        self.critic_fc_2 = torch.nn.Sequential(
            torch.nn.Linear(128+action_dim, 128),
            nn.ReLU(inplace=True),
            torch.nn.Linear(128, 1)
        )

        self.apply(torch_utils.weights_init)

    def forward(self, obs, act):
        conv_out = self.state_conv_1(obs)
        out_1 = self.critic_fc_1(torch.cat((conv_out, act), dim=1))
        out_2 = self.critic_fc_2(torch.cat((conv_out, act), dim=1))
        return out_1, out_2
    


class SACCriticCQL(nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5):
        super().__init__()
        print("Non-Equi Q Network")
        self.state_conv_1 = SACEncoder(obs_shape, 128)

        # Q1
        self.critic_fc_1 = torch.nn.Sequential(
            torch.nn.Linear(128+action_dim, 128),
            nn.ReLU(inplace=True),
            torch.nn.Linear(128, 1)
        )

        # Q2
        self.critic_fc_2 = torch.nn.Sequential(
            torch.nn.Linear(128+action_dim, 128),
            nn.ReLU(inplace=True),
            torch.nn.Linear(128, 1)
        )

        self.apply(torch_utils.weights_init)

    def forward(self, obs, act):
        conv_out = self.state_conv_1(obs)
        out_1 = self.critic_fc_1(torch.cat((conv_out, act), dim=1))
        out_2 = self.critic_fc_2(torch.cat((conv_out, act), dim=1))
        return out_1, out_2
    

    def forward_features(self, obs):
        return self.state_conv_1(obs)
    

    def forward_critic_head(self, features, act):
        out_1 = self.critic_fc_1(torch.cat((features, act), dim=1))
        out_2 = self.critic_fc_2(torch.cat((features, act), dim=1))
        return out_1, out_2
    




# similar amount of parameters
class SACCriticGoal(nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5,n_tasks=2):
        super().__init__()
        print("Non-Equi Q Network")
        self.state_conv_1 = SACEncoder(obs_shape, 128)

        # Q1
        self.critic_fc_1 = torch.nn.Sequential(
            torch.nn.Linear(128+action_dim+n_tasks, 128),
            nn.ReLU(inplace=True),
            torch.nn.Linear(128, 1) 
        )

        # Q2
        self.critic_fc_2 = torch.nn.Sequential(
            torch.nn.Linear(128+action_dim+n_tasks, 128),
            nn.ReLU(inplace=True),
            torch.nn.Linear(128, 1)
        )

        self.apply(torch_utils.weights_init)

    def forward(self, obs, act, goal):
        conv_out = self.state_conv_1(obs)
        out_1 = self.critic_fc_1(torch.cat((conv_out, act, goal), dim=1))
        out_2 = self.critic_fc_2(torch.cat((conv_out, act, goal), dim=1))
        return out_1, out_2
    

    


# similar amount of parameters
class SACIQLValueCritic(nn.Module):
    def __init__(self, obs_shape=(2, 128, 128)):
        super().__init__()
        print("Non-Equi IQL Value V Network")
        self.state_conv_1 = SACEncoder(obs_shape, 128)

        # V value
        self.critic_fc_1 = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            torch.nn.Linear(128, 1)
        )

        self.apply(torch_utils.weights_init)

    def forward(self, obs):
        conv_out = self.state_conv_1(obs)
        out_1 = self.critic_fc_1(conv_out)
        return out_1
    



class SACIQLValueCriticGoal(nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), goal_dim=2, n_tasks=2):
        super().__init__()
        print("Non-Equi IQL Value V Network Goal")
        self.n_tasks = n_tasks

        self.state_conv_1 = SACEncoder(obs_shape, 128)

        # V value
        self.critic_fc_1 = torch.nn.Sequential(
            torch.nn.Linear(128 + n_tasks, 128),
            nn.ReLU(inplace=True),
            torch.nn.Linear(128, 1)
        )

        self.apply(torch_utils.weights_init)

    def forward(self, obs, goal):
        conv_out = self.state_conv_1(obs)
        out_1 = self.critic_fc_1(torch.cat((conv_out, goal), dim=1))
        return out_1
    

    

# similar amount of parameters
class SACGaussianPolicy(SACGaussianPolicyBase):

    def __init__(self, obs_shape=(2, 128, 128), action_dim=5):
        super().__init__()
        self.conv = SACEncoder(obs_shape, 128)
        self.mean_linear = nn.Linear(128, action_dim)
        self.log_std_linear = nn.Linear(128, action_dim)

        self.apply(torch_utils.weights_init)

    def forward(self, x):
        x = self.conv(x)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std
    



class SACGaussianPolicyCQL(SACGaussianPolicyBase):

    def __init__(self, obs_shape=(2, 128, 128), action_dim=5):
        super().__init__()
        self.conv = SACEncoder(obs_shape, 128)
        self.mean_linear = nn.Linear(128, action_dim)
        self.log_std_linear = nn.Linear(128, action_dim)

        self.apply(torch_utils.weights_init)

    def forward(self, x):
        x = self.conv(x)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std
    
    def forward_features(self, x):
        x = self.conv(x)
        return x
    
    def forward_actor_head(self, features):
        mean = self.mean_linear(features)
        log_std = self.log_std_linear(features)
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

    


class SACGaussianPolicyGoal(SACGaussianPolicyBase):

    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, goal_dim=2, n_tasks=2):
        super().__init__()
        self.n_tasks = n_tasks

        self.conv = SACEncoder(obs_shape, 128)
        self.mean_linear = nn.Linear(128 + n_tasks, action_dim)
        self.log_std_linear = nn.Linear(128 + n_tasks, action_dim)

        self.apply(torch_utils.weights_init)
    

    def forward(self, x, goal):

        x = self.conv(x)
        x = torch.cat((x, goal), dim=1)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std
    





class SACIQLPolicy(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5):
        super().__init__()
        print("Non-Equi IQL Policy")
        self.conv = SACEncoder(obs_shape, 128)
        self.mean_linear = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(-1 * torch.ones(action_dim, requires_grad=True))

        self.apply(torch_utils.weights_init)



    def forward(self, x):
        batch_size = x.shape[0]

        x = self.conv(x)
        mean = self.mean_linear(x)

        log_std_expanded = self.log_std.expand(batch_size, -1)
        log_std = torch.clamp(log_std_expanded, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std





class SACIQLPolicyGoal(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, goal_dim=2, n_tasks=2):
        super().__init__()
        print("Non-Equi IQL Policy Goal")
        self.n_tasks = n_tasks

        self.conv = SACEncoder(obs_shape, 128)
        self.mean_linear = nn.Linear(128 + n_tasks, action_dim)
        self.log_std = nn.Parameter(-1 * torch.ones(action_dim, requires_grad=True))

        self.apply(torch_utils.weights_init)

    def forward(self, x, goal):
        batch_size = x.shape[0]

        x = self.conv(x)
        x = torch.cat((x, goal), dim=1)
        mean = self.mean_linear(x)

        log_std_expanded = self.log_std.expand(batch_size, -1)
        log_std = torch.clamp(log_std_expanded, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std
    
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





class SACGaussianPolicy_2_Objects(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(16,), action_dim=5):
        super().__init__()

        obs_dims = obs_shape[0]
        self.linear_1 = nn.Linear(obs_dims, 128)
        self.linear_2 = nn.Linear(128, 128)
        self.mean_linear = nn.Linear(128, action_dim)
        self.log_std_linear = nn.Linear(128, action_dim)

        self.apply(torch_utils.weights_init)

    def forward(self, x):

        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std
    

class SACCritic_2_Objects(nn.Module):
    def __init__(self, obs_shape=(16,), action_dim=5):
        super().__init__()


        obs_dims = obs_shape[0]
        self.linear_1 = nn.Linear(obs_dims + action_dim, 128)
        self.linear_2 = nn.Linear(128, 128)

        self.state_conv_1 = SACEncoder(obs_shape, 128)

        # Q1
        self.critic_fc_1 = torch.nn.Sequential(
            torch.nn.Linear(128+action_dim, 128),
            nn.ReLU(inplace=True),
            torch.nn.Linear(128, 1)
        )

        # Q2
        self.critic_fc_2 = torch.nn.Sequential(
            torch.nn.Linear(128+action_dim, 128),
            nn.ReLU(inplace=True),
            torch.nn.Linear(128, 1)
        )

        self.apply(torch_utils.weights_init)

    def forward(self, obs, act):
        x = torch.cat((obs, act), dim=1)
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))

        out_1 = self.critic_fc_1(torch.cat((x, act), dim=1))
        out_2 = self.critic_fc_2(torch.cat((x, act), dim=1))
        return out_1, out_2