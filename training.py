from time import time
import numpy as np


def rollout_trajectory(env, algo):
    lstm = False
    if algo.config['model']['use_lstm']:
        lstm = True
        state = np.zeros((2, algo.config['model']['lstm_cell_size']))  # init state to all zeros
    obs = env.vector_reset()
    env.render()
    done = [False]
    total_reward = 0.0
    while not np.array(done).all():
        if lstm:  # pass the state as well
            action, state, _ = algo.compute_single_action(obs[0], state)
        else:
            action = algo.compute_single_action(obs[0])
        obs, reward, done, info = env.vector_step([action])
        env.render()
        total_reward += reward[0]
    print(f"Rollout total-reward={total_reward}")


def train(algo, eval_env, num_epochs, model_dir='models', eval_rollouts=0, checkpoint_ep=10):
    for ep in range(num_epochs):
        start = time()
        results = algo.train()
        print("Epoch {:d} took {:.2f} seconds; avg. reward={:.3f}".format(ep + 1, (time() - start),
                                                                          results['episode_reward_mean']))
        if (ep + 1) % checkpoint_ep == 0:
            print("Saving checkpoint to {}".format(algo.save(model_dir + 'checkpoints')))  # save checkpoint
            for _ in range(eval_rollouts):
                # rollout a trajectory using the learned model
                rollout_trajectory(eval_env, algo)
