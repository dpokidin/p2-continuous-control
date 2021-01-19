import numpy as np
import random
import copy
from model_plag import Actor, Critic
import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.97            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 3e-4         # learning rate of the actor 
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 1        # updates local networks every n steps
BSTP = 3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed, n_agents = 1 ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            n_agents (int) : number of agents
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.t_step = 0
        self.n_agents = n_agents
        self.buffer_size = int(BUFFER_SIZE/self.n_agents)
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise((n_agents,action_size), random_seed)

        # Replay memory
        self.memory = ReplayBuffer(state_size, action_size, 
                                   int(self.buffer_size*self.n_agents), 
                                   BATCH_SIZE, random_seed, n_agents=self.n_agents)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states = experiences['state'][:,:,0]
        actions = experiences['action'][:,:,0]
        rewards = experiences['reward'].sum(dim=2)
        next_states = experiences['next_state'][:,:,-1]
        dones = experiences['done'][:,:,-1] 

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
            
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        


        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1):
        """Initialize parameters and noise process."""
        self.shape = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self, sigma=None):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.normal(size=self.shape)
        self.state = x + dx
        return self.state

class Data:
    def __init__(self, shape, dtype, n_agents=1):
        self.N = shape[0]
        self.data = np.empty(shape, dtype)
        self.row = 0
        shape = (n_agents, ) + shape[-2:]
        self.sequence = Sequence(shape) 
        self.n_agents = n_agents
        
        
    def get(self):
        return self.data[:self.row]
        
    def retrieve(self, idx):
        data = self.get()
        return data[idx]
    
    def add(self, value):
        self.sequence(value)
        if self.sequence.isfull():
            if self.row < self.N:
                self.data[self.row:self.row+self.n_agents] = self.sequence.data
                self.row += self.n_agents
            else:
                self.data[:-self.n_agents] = self.data[self.n_agents:]
                self.data[-self.n_agents:] = self.sequence.data
            self.sequence.reset()
            
    def update(self, value, idx):
        self.data[idx] = value
        
class Sequence:
    def __init__(self, shape):
        self.shape = shape
        self.reset()
        self.N = shape[-1]
        
    def __call__(self, data):
        self.data[:, :, self.n] = data
        self.n +=1
        if self.n==self.N:
            self.full=True
            
    def isfull(self):
        return self.full
    
    def reset(self):
        self.data = np.empty(self.shape)
        self.n = 0
        self.full = False
        
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, state_size, action_size, buffer_size, batch_size, seed, n_agents):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.experience = {"state" : Data( (buffer_size, state_size, BSTP), np.float32, n_agents=n_agents),
                           "action": Data( (buffer_size, action_size, BSTP), np.float32, n_agents=n_agents), 
                           "reward": Data( (buffer_size, 1, BSTP), np.float32, n_agents=n_agents), 
                           "next_state" : Data( (buffer_size, state_size, BSTP), np.float32, n_agents=n_agents), 
                           "done" : Data( (buffer_size, 1, BSTP), np.float32, n_agents=n_agents)}
        
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.experience["state"].add(state)
        self.experience["action"].add(action)
        self.experience["reward"].add(reward)
        self.experience["next_state"].add(next_state)
        self.experience["done"].add(done)
        
    
    def sample(self):
        """Randomly sample a batch of experiences from memory according to piorities."""
        idx = np.random.choice(np.arange(len(self)), self.batch_size, replace=False)
        output = {}
        data = self.experience['state'].retrieve(idx)
        output['state']      = torch.from_numpy(data).float().to(device)
        data = self.experience['action'].retrieve(idx)
        output['action']     = torch.from_numpy(data).float().to(device)
        data = self.experience['reward'].retrieve(idx)
        output['reward']     = torch.from_numpy(data).float().to(device)
        data = self.experience['next_state'].retrieve(idx)
        output['next_state'] = torch.from_numpy(data).float().to(device)
        data = self.experience['done'].retrieve(idx)
        output['done'] = torch.from_numpy(data).float().to(device)
        return output

    def __len__(self):
        """Return the current size of internal memory."""
        return self.experience['state'].row