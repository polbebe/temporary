from control.RL.SAC import SAC
from body.PinkPanther.PinkPantherEnv import EasyPinkPantherEnv, HardPinkPantherEnv, RandPinkPantherEnv, PinkPantherEnv, ConstPinkPantherEnv, RandConstPinkPantherEnv
import numpy as np
from collections import defaultdict
from collections import deque
import time
import json

def get_envs(Env, E, node_id, env_str, data_path):
    ids = [int(node[1].split('_')[1]) for node in E]
    ids = [i for i in ids if i < node_id]
    if len(ids) == 0:
        train_envs = [test_env]
    else:
        train_envs = []
        for i in ids:
            env = Env()
            env.load_config(data_path + '/' + env_str + '/' + env_str + '_train_config_' + str(i))
            train_envs.append(env)
    return train_envs

def get_dataset(env, train_episodes=100):
    i = 0
    train = []
    episode_step = 0
    obs = env.reset()
    while i < train_episodes:
        action = np.random.uniform(-1, 1, env.action_space.shape[0])
        new_obs, r, done, info = env.step(action)
        train.append([obs, action, r, new_obs, done, episode_step])
        episode_step += 1
        obs = new_obs
        if done:
            episode_step = 0.0
            obs = env.reset()
            i += 1
    return train

class Nearest:
    def __init__(self, env, model_path):
        self.env = env
        self.dataset = get_dataset(env)
        self.model_path = model_path

    def test_dist(self, a, b, dev='cuda'):
        from utils import test_valid
        import torch
        env_b = b.split("_")[0]
        robot_b = b.split("_")[1]
        model_b = torch.load(self.model_path + env_b + '_model_' + robot_b + '.pt', map_location='cuda:0').to(dev)

        # data_a = np.load(data_path + a +'.npy', allow_pickle=True)[-1000:]
        edge_wt = test_valid(model_b, self.dataset)
        return edge_wt

    def find_nearest(self, node_id, path, k):
        from traversal.Greedy import GreedyTraversal
        oldEdgeDict = json.load(open(path))
        G = list(oldEdgeDict.keys())[:node_id]
        S = set(G)

        # cleaning
        edgeDict = dict()
        for key in oldEdgeDict:
            if key in S:
                edgeDict[key] = []
                for e, sub_key in oldEdgeDict[key]:
                    if sub_key in S:
                        edgeDict[key].append([e, sub_key])

        algo = GreedyTraversal(G, edgeDict, self.test_dist)
        E, W = algo.addNode('', k=k)
        return E

def find_random(node_id, path, k):
    oldEdgeDict = json.load(open(path))
    G = list(oldEdgeDict.keys())
    r = np.random.randint(0, len(G), k)
    E = [('', G[i]) for i in r]
    return E

class MultiEnv:
    def __init__(self, envs):
        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self, *kwargs):
        idx = np.random.randint(0, len(self.envs), 1)[0]
        self.chosen_env = self.envs[idx]
        return self.chosen_env.reset(*kwargs)

    def step(self, *kwargs):
        return self.chosen_env.step(*kwargs)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--node', type=int, default=100)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--log', type=str, default='rl_logs/')
    parser.add_argument('--graph', type=str, default='PinkPanther_rand_greedy_5_1000.json')
    parser.add_argument('--env', type=str, default='rand')
    args = parser.parse_args()
    node_id = args.node
    if args.env == 'rand':
        test_env = PinkPantherEnv()
        Env = RandPinkPantherEnv
    elif args.env == 'randConst':
        test_env = ConstPinkPantherEnv(PinkPantherEnv())
        Env = RandConstPinkPantherEnv
    else:
        test_env = PinkPantherEnv()
        Env = RandPinkPantherEnv
    if node_id == 0:
        print('Running Baseline')
        train_env = test_env
    if node_id == 1:
        E = find_random(args.node, args.graph, args.k)
        train_envs = get_envs(Env, E, node_id, args.env, '/data/PinkPanther')
        train_env = MultiEnv(train_envs)
    else:
        E = Nearest(test_env, 'trained/PinkPanther_'+args.env+'_greedy_5_1000/').find_nearest(args.node, args.graph, args.k)
        train_envs = get_envs(Env, E, node_id, args.env, '/data/PinkPanther')
        train_env = MultiEnv(train_envs)
    sac = SAC(train_env, log_dir=args.log+args.env+'_'+str(node_id))
    sac.learn(1000000, test_env=test_env)
    print('Done')
