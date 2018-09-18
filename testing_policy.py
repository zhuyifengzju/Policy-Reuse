from maze import Maze
from pi_reuse import Policy
from pathlib import Path
import argparse
import numpy as np

def arg_parse():
    parser = argparse.ArgumentParser(description='Policy Reuse for maze')
    parser.add_argument(
        '-o',
        '--output',
        default='TRAINED_POLICY',
        type=Path,
        help='Directory to save data to.')

    parser.add_argument(
        '--task-id', default=1, type=int, help='Episode length')

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    
    return args
    


def main():
    args = arg_parse()
    
    env = Maze(
        height=21,
        width=24,
        scale=20,
        channel=3,
        mask_fn='identity',
        use_image_action=False,
        random_init_pos=True,
        use_discrete_state=True,
        use_grey_image=False,
        task_id=args.task_id)
    """
    Initialize policy
    """
    # random_policy = Policy(state_shape=env.observation_space.shape,
    #                        action_shape=env.action_space.n,
    #                        task_id=args.task_id,
    #                        policy_name='random',
    #                        output_dir=args.output)

    # random_policy.train(env, args.k, args.h, args.gamma, args.alpha)

    policy_name = 'e-greedy'
    for i in range(1):
        
        filename = args.output / f'{policy_name}-2000.npy'
        policy, w = Policy.load(filename)
        is_terminal = False
        state = env.reset()
        while is_terminal is False:
            action = np.argmax(policy[state[0], state[1], :])
            print(policy[state[0], state[1], :])
            next_state, reward, is_terminal, _ = env.step(action)
            env.render()
            state = next_state
            input()
if __name__  == '__main__':
    main()
