from time import time
import json
import numpy as np


def load_model_config(model_dir):
    with open(model_dir + 'model_config.json', 'r') as f:
        model_config = json.load(f)
    return model_config


def save_model_config(model_dir, model_config):
    with open(model_dir + 'model_config.json', 'w') as f:
        json.dump(model_config, f, indent=4)


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


def train(algo, num_epochs, model_dir='models', checkpoint_ep=10):
    for ep in range(num_epochs):
        start = time()
        results = algo.train()
        mean_action_reward = np.sum(results['hist_stats']['episode_reward']) / np.sum(results['hist_stats']['episode_lengths'])
        print("Epoch {:d} took {:.2f} seconds; avg. episode reward={:.3f}, avg. episode length={:.2f}, avg. action reward={:.2f}".format(ep + 1, results['time_this_iter_s'],
                                                                          results['episode_reward_mean'], results['episode_len_mean'], mean_action_reward))
        algo.evaluate()
        if (ep + 1) % checkpoint_ep == 0:
            print("Saving checkpoint to {}".format(algo.save(model_dir + 'checkpoints')))  # save checkpoint
            # for _ in range(eval_rollouts):
                # rollout a trajectory using the learned model
                # rollout_trajectory(eval_env, algo)
