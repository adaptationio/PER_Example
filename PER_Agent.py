
class WeightedLoss(nn.Module):
    def __init__(self):
        super(WeightedLoss, self).__init__()
        return

    def forward(self, weights, inputs, targets):
        t = torch.abs(inputs - targets)
        zi = torch.where(t < 1, 0.5 * t ** 2, t - 0.5)
        loss = (weights * zi).sum()
        return loss

class PERDQNAgent(object):
    
    def __init__(self, state_size, action_size, seed, hidden_layer_sizes = [128, 128, 128]):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.e = 1e-1
        
        self.network_local = DQNetwork(state_size, action_size, seed, hidden_layer_sizes).to(device)
        self.network_target = DQNetwork(state_size, action_size, seed, hidden_layer_sizes).to(device)
        print("Local Netwrok: ")
        print(self.network_local)
        print("Target Netwrok: ")
        print(self.network_target)
        self.optimizer = optim.Adam(self.network_local.parameters(), lr=LR)
        
        # Prioritized Replay memory
        self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.criterion = WeightedLoss()
        
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        
        self.memory.add(state, action, reward, next_state, done)
        
        
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(b=1)
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.network_local.eval()
        with torch.no_grad():
            action_values = self.network_local(state)
        self.network_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        idxes, states, actions, rewards, next_states, is_weights, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.network_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.network_local(states).gather(1, actions)
        
        # Get priorities and update
        priorities = torch.abs(Q_targets - Q_expected) + 1.
        
        priorities = priorities.squeeze(1).cpu().data.numpy()
        self.memory.update(idxes, priorities)
        
        # Compute loss
        self.optimizer.zero_grad()
        loss = self.criterion(is_weights, Q_expected, Q_targets)
        # Minimize the loss    
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.network_local, self.network_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)