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