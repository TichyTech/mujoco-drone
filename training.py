from time import time
import numpy as np


def rollout_trajectory(env, algo):
    obs = env.vector_reset()
    env.render()
    done = [False]
    total_reward = 0.0
    while not np.array(done).all():
        action = algo.compute_single_action(obs[0])
        obs, reward, done, info = env.vector_step([action])
        env.render()
        total_reward += reward[0]
    print(f"Rollout total-reward={total_reward}")


def train(algo, eval_env, num_epochs, eval_rollouts=0, model_dir='models'):
    for ep in range(num_epochs):
        start = time()
        results = algo.train()
        print("Epoch {:d} took {:.2f} seconds; avg. reward={:.3f}".format(ep + 1, (time() - start),
                                                                          results['episode_reward_mean']))
        if (ep + 1) % 10 == 0:
            print("Saving checkpoint to {}".format(algo.save(model_dir + 'checkpoints')))  # save checkpoint
            for _ in range(eval_rollouts):
                # rollout a trajectory using the learned model
                rollout_trajectory(eval_env, algo)
