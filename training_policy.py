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
        '--seed', default=0, type=int, help='Random Seed')

    parser.add_argument(
        '-g', '--gamma', default=0.95, type=float, help='Coefficient gamma')
    parser.add_argument(
        '-a', '--alpha', default=0.05, type=float, help='Coefficient gamma')

    parser.add_argument(
        '--k', default=2000, type=int, help='Number of episodes')
    parser.add_argument(
        '--h', default=100, type=int, help='Episode length')

    parser.add_argument(
        '--task-id', default=1, type=int, help='Episode length')

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
    """
    Initialize policy
    """
    # random_policy = Policy(state_shape=env.observation_space.shape,
    #                        action_shape=env.action_space.n,
    #                        task_id=args.task_id,
    #                        policy_name='random',
    #                        output_dir=args.output)

    # random_policy.train(env, args.k, args.h, args.gamma, args.alpha)

    # epsilon_greedy_policy = Policy(state_shape=env.observation_space.shape,
    #                                action_shape=env.action_space.n,
    #                                task_id=args.task_id,
    #                                policy_name='e-greedy',
    #                                output_dir=args.output,
    #                                seed=args.seed)

    # epsilon_greedy_policy.train(env, args.k, args.h, args.gamma, args.alpha)

    boltzmann_policy = Policy(state_shape=env.observation_space.shape,
                                   action_shape=env.action_space.n,
                                   task_id=args.task_id,
                                   policy_name='boltzmann',
                                   output_dir=args.output,
                                   seed=args.seed)

    boltzmann_policy.train(env, args.k, args.h, args.gamma, args.alpha)

    # greedy_policy = Policy(state_shape=env.observation_space.shape,
    #                                action_shape=env.action_space.n,
    #                                task_id=args.task_id,
    #                                policy_name='greedy',
    #                                output_dir=args.output,
    #                                seed=args.seed)

    # greedy_policy.train(env, args.k, args.h, args.gamma, args.alpha)

if __name__  == '__main__':
    main()
# 
