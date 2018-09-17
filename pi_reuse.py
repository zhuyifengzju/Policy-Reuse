from maze import Maze

def main():

    env = Maze(
        height=21,
        width=24,
        scale=20,
        channel=3,
        mask_fn='identity',
        use_image_action=False,
        # use_discrete_state=True,
        use_grey_image=False)

    state = env.reset()
    env.render()
    while True:
        k = int(input())
        next_state, _, _, _ = env.step(k)
        env.render()


if __name__  == '__main__':
    main()
