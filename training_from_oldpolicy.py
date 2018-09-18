from maze import Maze
from pi_reuse import Policy, PiReuseAgent
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
        '--task-id', default=1, type=int, help='Episode length')

    parser.add_argument(
        '--old-id', default=1, type=int, help='Episode length')

    parser.add_argument(
        '--load-policy', default=None, type=Path, help='Episode length')

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



    reuse_policy = PiReuseAgent(state_shape=env.observation_space.shape,
                                action_shape=env.action_space.n,
                                task_id=args.task_id,
                                old_id=args.old_id,
                                policy_name='pi-reuse',
                                output_dir=args.output,
                                seed=args.seed)
    old_policy, _ = Policy.load(f'{args.load_policy}/e-greedy-2000.npy')
    reuse_policy.train(env, old_policy,
                       args.k, args.h, args.gamma, args.alpha, args.psi, args.miu)
    
if __name__  == '__main__':
    main()
# 
