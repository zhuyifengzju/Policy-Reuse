"""Policy Reuse Q learning"""
import numpy as np
from typing import List, Tuple

class PolicyReuseQLAgent():
    def __init__(
            self,
            state_shape: Tuple,
            action_shape: Tuple,
            task_id=0,
            policy_library: List=[],
            policy_name='prql',
            output_dir=None,
            seed=0,
    ):
        """
        state_shape: shape of state
        action_shape: shape of action
        task_id: Current task id
        policy_library: a list of poclies
        """
        self.state_shape = state_shape
        if isinstance(action_shape, int):
            action_shape = (action_shape, )

        self.action_shape = action_shape

        self.task_id = task_id
        self.policy_name = policy_name
        self.output_dir = output_dir / f'{policy_name}' / f'{task_id}'

        self.policy_library = policy_library
        self.num_policies = len(policy_library) + 1
        # Index of the policy going to reuse
        
        np.random.seed(seed)
        #1
        self.policy = np.zeros(state_shape + action_shape)
        #2,3
        self.reuse_gains = np.zeros(self.num_policies)
        #4,5
        self.num_chosen = np.zeros(self.num_policies)
        
        self.global_step = 0
        
    def train(self, env, K, H, gamma, alpha, psi, miu):
        total_reward = 0
        
        for k in range(K):
            reuse_idx = self.choose_from_pl()
            
            
            if reuse_idx==0:
                policy_fn = self.greedy
                policy = self.policy
            else:
                policy_fn = self.policy_reuse
                policy = self.policy_library[reuse_idx-1]

            episode_reward = 0
            w = 0
            w_gamma = 1
            self.psi = psi

                        
            state = env.reset()
            i = 0
            is_terminal = False
            episode_reward = 0
            while is_terminal is False and i < H:
                i += 1
                action = policy_fn(policy, *state)
                next_state, reward, is_terminal, debug_info = env.step(action)
                env.render()
                # Update policy
                x, y = state[0], state[1]
                x_prime, y_prime = next_state[0], next_state[1]
                self.policy[x, y, action] = ((1-alpha) * self.policy[x, y, action]
                                              + alpha * (reward + gamma * np.max(self.policy[x_prime, y_prime, :])))
                self.psi = self.psi * miu
                episode_reward  = reward + gamma * episode_reward
                w = w + w_gamma * reward
                w_gamma = w_gamma * gamma
                
                state = next_state

            self.global_step += 1
            # Update parameters
            self.reuse_gains[reuse_idx] = ((self.reuse_gains[reuse_idx] * self.num_chosen[reuse_idx] + w)
                                           / (self.num_chosen[reuse_idx] + 1))
            self.num_chosen[reuse_idx] = self.num_chosen[reuse_idx] + 1
            # tau is calculated directly in choose_from_pl()
            
            total_reward += w
            if (k % 100 == 99 or k == 0):
                print(f'Iteration{k+1}: {self.policy_name} - {total_reward / self.global_step}, Reuse gains: {self.reuse_gains}')
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
    
    def policy_reuse(self, old_policy, x, y):
        rdn = np.random.uniform(0, 1)
        if rdn < self.psi:
            return self.greedy(old_policy, x, y)
        else:
            return self.epsilon_greedy(self.policy, x, y)
        
    def greedy(self, policy, x, y):
        return np.argmax(policy[x, y, :])
        
    def epsilon_greedy(self, policy, x, y):
        e = np.random.uniform(0, 1)
        if e < self.psi:
            return np.random.randint(self.action_shape[0])
        else:
            return np.argmax(policy[x, y, :])

    def choose_from_pl(self):
        """
        Return an index from policy library.
        if index is 0: use Pi_{\omega}
        else use Pi_{index}
        """
        tau = self.global_step * 0.05
        exponent = np.exp(self.reuse_gains * tau)
        logits = exponent / np.sum(exponent)

        return np.random.choice(self.num_policies, p = logits)
