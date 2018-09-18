from pathlib import Path
from typing import Tuple
import numpy as np

class Policy():
    def __init__(
            self,
            state_shape: Tuple,
            action_shape: Tuple,
            task_id=0,
            policy_name='e-greedy',
            output_dir=None,
            seed=0,
    ):
        self.state_shape = state_shape
        if isinstance(action_shape, int):
            action_shape = (action_shape, )

        self.action_shape = action_shape

        self.task_id = task_id
        self.policy_name = policy_name
        self.output_dir = output_dir / f'{policy_name}' / f'{task_id}-{seed}'

        np.random.seed(seed)
        self.policy = np.zeros(state_shape + action_shape)

        self.global_step = 0
        if policy_name == 'random':
            self.policy_fn = self.random
        elif policy_name == 'greedy':
            self.policy_fn = self.greedy
        elif policy_name == 'e-greedy':
            self.policy_fn = self.epsilon_greedy
        elif policy_name == 'boltzmann':
            self.policy_fn = self.boltzmann

        
    def train(self, env, K, H, gamma, alpha):
        total_reward = 0
        for k in range(K):

            episode_reward = 0
            w = 0
            w_gamma = 1
            state = env.reset()
            i = 0
            is_terminal = False
            while is_terminal is False and i < H:
                i += 1
                action = self.policy_fn(*state)
                next_state, reward, is_terminal, debug_info = env.step(action)
                env.render()
                # Update policy
                x, y = state[0], state[1]
                x_prime, y_prime = next_state[0], next_state[1]
                self.policy[x, y, action] = ((1-alpha) * self.policy[x, y, action]
                                              + alpha * (reward + gamma * np.max(self.policy[x_prime, y_prime, :])))
                # episode_reward  = reward + gamma * episode_reward
                w = w + w_gamma * reward
                w_gamma = w_gamma * gamma
                state = next_state
            self.global_step += 1

            # total_reward += episode_reward
            total_reward += w
            if (k % 100 == 99 or k == 0):
                print(f'Iteration{k+1}: {self.policy_name} - {total_reward / self.global_step}')
                self.save(self.output_dir, total_reward / self.global_step)
        
        
    def save(self, output_dir, W):
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'{self.policy_name}-{self.global_step}.npy'
        dict = {
            'policy': self.policy,
            'W': W
        }
        np.save(output_file, dict)

    @staticmethod
    def load(load_file):

        load_array = np.load(load_file)
        return load_array.item().get('policy'), load_array.item().get('W')
        
    def random(self, x, y):
        return np.random.randint(self.action_shape[0])

    def greedy(self, x, y):
        return np.argmax(self.policy[x, y, :])
    
    def epsilon_greedy(self, x, y):
        e = np.random.uniform(0, 1)
        if e < 1 - (self.global_step-1) * 0.0005:
            return np.random.randint(self.action_shape[0])
        else:
            return np.argmax(self.policy[x, y, :])
            
    def boltzmann(self, state):
        # tau = self.global_step * 5
        # exponent = np.exp(self.policy * tau)

        # logits = exponent / np.sum(exponent)
        # return np.random.multinomial(self.action_shape, logits)[0]
        pass
    

class PiReuseAgent():
    def __init__(
            self,
            state_shape: Tuple,
            action_shape: Tuple,
            task_id=0,
            old_id=0,
            policy_name='pi-reuse',
            output_dir=None,
            seed=0,
    ):
        self.state_shape = state_shape
        if isinstance(action_shape, int):
            action_shape = (action_shape, )

        self.action_shape = action_shape

        self.task_id = task_id
        self.policy_name = policy_name
        self.output_dir = output_dir / f'{policy_name}' / f'{task_id}-learn-from-reuse-{old_id}'

        np.random.seed(seed)
        self.policy = np.zeros(state_shape + action_shape)

        self.global_step = 0
        self.policy_fn = self.policy_reuse
        
    def train(self, env, old_policy, K, H, gamma, alpha, psi, miu):
        total_reward = 0
        self.old_policy = old_policy
        for k in range(K):

            episode_reward = 0
            w = 0
            w_gamma = 1
            state = env.reset()
            i = 0
            is_terminal = False
            self.psi = psi
            while is_terminal is False and i < H:
                i += 1
                action = self.policy_fn(*state)
                next_state, reward, is_terminal, debug_info = env.step(action)
                env.render()
                # Update policy
                x, y = state[0], state[1]
                x_prime, y_prime = next_state[0], next_state[1]
                self.policy[x, y, action] = ((1-alpha) * self.policy[x, y, action]
                                              + alpha * (reward + gamma * np.max(self.policy[x_prime, y_prime, :])))
                self.psi = self.psi * miu
                # episode_reward  = reward + gamma * episode_reward
                w = w + w_gamma * reward
                w_gamma = w_gamma * gamma
                
                state = next_state
            self.global_step += 1

            # total_reward += episode_reward
            total_reward += w
            if (k % 100 == 99 or k == 0):
                print(f'Iteration{k+1}: {self.policy_name} - {total_reward / self.global_step}')
                self.save(self.output_dir, total_reward / self.global_step)
        
        
    def save(self, output_dir, W):
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'{self.policy_name}-{self.global_step}.npy'
        dict = {
            'policy': self.policy,
            'W': W
        }
        np.save(output_file, dict)

    @staticmethod
    def load(load_file):

        load_array = np.load(load_file)
        return load_array.item().get('policy'), load_array.item().get('W')
    
    def policy_reuse(self, x, y):
        psi = np.random.uniform(0, 1)
        if psi < self.psi:
            return np.argmax(self.old_policy[x, y, :])
        else:
            return self.epsilon_greedy(x, y)
    
    def epsilon_greedy(self, x, y):
        e = np.random.uniform(0, 1)
        if e < self.psi:
            return np.random.randint(self.action_shape[0])
        else:
            return np.argmax(self.policy[x, y, :])
