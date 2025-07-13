import random
from collections import deque

import torch
import numpy as np

from dqn import DeepQNetwork


class Agent():
    def __init__(
        self, 
        input_dims: tuple[int, int],
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        eps_min: float = 0.01,
        eps_dec: float = 0.995,
        mem_size: int = 1000,
        batch_size: int = 64,
        target_update: int = 1000
        ) -> None:

        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.lr = lr
        self.batch_size = batch_size
        self.target_update = target_update
        self.learn_step_counter = 0
        self.memory: deque = deque(maxlen=mem_size)

        self.q_eval = DeepQNetwork(lr, input_dims, 128, 128, n_actions)
        self.q_target = DeepQNetwork(lr, input_dims, 128, 128, n_actions)
        self.q_target.load_state_dict(self.q_eval.state_dict())
        self.q_target.eval()

        self.n_actions = n_actions

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
        ) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return np.random.choice(self.n_actions)

        state_tensor = torch.tensor([state], dtype=torch.float).to(self.q_eval.device)
        actions = self.q_eval.forward(state_tensor)
        return torch.argmax(actions).item()

    def learn(self) -> None:
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        states_tensor = torch.tensor(states, dtype=torch.float32).to(self.q_eval.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.q_eval.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.q_eval.device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(self.q_eval.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).to(self.q_eval.device)
        
        q_pred = self.q_eval(states_tensor).gather(1, actions_tensor).squeeze()
        q_next = self.q_target(next_states_tensor).max(dim=1)[0]
        q_target = rewards_tensor + self.gamma * q_next * (1 - dones_tensor)

        loss = self.q_eval.loss(q_pred, q_target)

        self.q_eval.optimizer.zero_grad()
        loss.backward()
        self.q_eval.optimizer.step()

        self.learn_step_counter += 1

        if self.learn_step_counter % self.target_update == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())

        self.epsilon = max(self.eps_min, self.epsilon * self.eps_dec)
