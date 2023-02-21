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
    start = time()
    for ep in range(num_epochs):
        results = algo.train()
        mean_action_reward = np.sum(results['hist_stats']['episode_reward']) / np.sum(results['hist_stats']['episode_lengths'])
        elapsed = int(time() - start)
        print("Elapsed time: {:3d}h {:2d}m; ep. {:4d}, avg.e.r.={:.3f}, avg. l={:.2f}, avg.a.r.={:.2f}".format(elapsed//3600, (elapsed//60)%60, ep + 1,
              results['episode_reward_mean'], results['episode_len_mean'], mean_action_reward))
        algo.evaluate()
        if (ep + 1) % checkpoint_ep == 0:
            print("Saving checkpoint to {}".format(algo.save(model_dir + 'checkpoints')))  # save checkpoint