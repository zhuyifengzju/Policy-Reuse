from maze import Maze
from pi_reuse import Policy
from prql import PolicyReuseQLAgent
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
        '-i',
        '--input',
        default='TRAINED_POLICY',
        type=Path,
        help='Directory of reusing data.')

    parser.add_argument(
        '--seed', default=0, type=int, help='Random Seed')

    parser.add_argument(
        '-g', '--gamma', default=0.95, type=float, help='Coefficient gamma')
    parser.add_argument(
        '-a', '--alpha', default=0.05, type=float, help='Coefficient gamma')
    parser.add_argument(
        '--psi', default=1.0, type=float, help='Coefficient gamma')
    parser.add_argument(
        '--miu', default=0.95, type=float, help='Coefficient gamma')

    parser.add_argument(
        '--k', default=2000, type=int, help='Number of episodes')
    parser.add_argument(
        '--h', default=100, type=int, help='Episode length')

    parser.add_argument(
        '--task-id', default=0, type=int, help='Episode length')

    parser.add_argument('-r','--reuse-id', nargs='+', type=int, help='Reuse id list', required=True)    

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    
    return args
    


def main():
    args = arg_parse()

    np.random.seed(args.seed)
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
        task_id=args.task_id,
        seed=args.seed)

    policy_library = []
    for i in args.reuse_id:
        filename = args.input / f'e-greedy/{i}-{args.seed}/e-greedy-2000.npy'
        policy, _ = Policy.load(filename)
        policy_library.append(policy)

    policy_name = 'prql'
    for i in args.reuse_id:
        policy_name = policy_name + f'-{i}'
    prql_agent = PolicyReuseQLAgent(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.n,
        task_id=args.task_id,
        policy_library=policy_library,
        policy_name=policy_name,
        output_dir=args.output,
        seed=args.seed)

    prql_agent.train(env, args.k, args.h, args.gamma, args.alpha, args.psi, args.miu)

if __name__  == '__main__':
    main()
# 
