from body.PinkPanther.PinkPantherEnv import EasyPinkPantherEnv, HardPinkPantherEnv, RandPinkPantherEnv
from control.RL.SAC import SAC
import json
import argparse
import gym
import random
import os
import numpy as np
import sys
import time
import pickle
from torch.multiprocessing import Process, Queue

def parallelize(inputs, P, Q):
    busy = [False for _ in range(len(P))]
    outputs = []
    idx = 0
    while idx < len(inputs):
        for i in range(len(busy)):
            # if any process is freed up add a new input there
            if busy[i] == False:
                Q[i][0].put(inputs[idx])
                busy[i] = True
                idx += 1
                if idx >= len(inputs): break
            # check at each step to see if a process is free
            # except and do nothing otherwise
            else:
                try:
                    v = Q[i][1].get_nowait()
                    outputs.append(v)
                    busy[i] = False
                except: pass

    # collect the stragglers
    for i in range(len(busy)):
        if busy[i] == True:
            v = Q[i][1].get()
            outputs.append(v)
            busy[i] = False

    assert len(outputs) == len(inputs)
    return outputs

class MultiEnv(gym.Env):
    def __init__(self, envs):
        self.envs = envs
        self.active = self.envs[0]
        self.action_space = self.active.action_space
        self.observation_space = self.active.observation_space

    def reset(self, *args):
        self.active = self.envs[random.randint(0, len(self.envs)-1)]
        return self.active.reset(*args)

    def step(self, *args):
        return self.active.step(*args)

def act(obs, t, a, b, c):
    current_p = obs[:12]
    desired_p = np.zeros(12)
    v = a * np.sin(t * b) + c
    pos = [1, 10, 2, 11]
    neg = [4, 7, 5, 8]
    zero = [0, 3, 6, 9]
    desired_p[pos] = v
    desired_p[neg] = -v
    desired_p[zero] = 0

    delta_p = 10 * (desired_p - current_p)
    delta_p = np.clip(delta_p, -1, 1)
    return delta_p

def test_params(params, envs, steps=100):
    ep_r = 0
    for env in envs:
        obs = env.reset()
        for t in range(steps):
            action = act(obs, t, *params)
            obs, r, done, info = env.step(action)
            ep_r += r
    return ep_r / len(envs)

def find_params(envs, episodes):
    params = np.random.uniform(-1, 1, (episodes, 4))
    params[:,0] = 0
    for i in range(len(params)):
        params[i,0] = test_params(params[i,1:], envs)
    p = np.argsort(params[:,0])[::-1]
    return params[p[0],1:]

def get_envs(node_id, env_str, graph_path, data_path):
    test_env = Env()
    test_env.load_config(data_path + '/' + env_str + '/' + env_str + '_train_config_' + str(node_id))

    G = json.load(open(graph_path+'/'+env_str+'_1.json'))
    ids = [int(node[1].split('_')[1]) for node in G[env_str+'_'+str(node_id)]]
    ids = [i for i in ids if i < node_id]
    if len(ids) == 0:
        train_envs = [test_env]
    else:
        train_envs = []
        for i in ids:
            env = Env()
            env.load_config(data_path + '/' + env_str + '/' + env_str + '_train_config_' + str(i))
            train_envs.append(env)
    return test_env, train_envs

def process(q_in, q_out, env_str, graph_path, data_path, episodes):
    id = q_in.get()
    np.random.seed(id)
    print('Process '+str(id)+' started.')
    while True:
        got = q_in.get()
        if got is None: break
        node_id = got
        test_env, train_envs = get_envs(node_id, env_str, graph_path, data_path)
        params = find_params(train_envs, episodes)
        # print(params)
        score = test_params(params, [test_env])
        # print(score)
        q_out.put(score)
    q_out.put(None)
    print('Process '+str(id)+' ended.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='easy')
    parser.add_argument('--graph-path', type=str, default='./')
    parser.add_argument('--data-path', type=str, default='/data/PinkPanther/')
    parser.add_argument('--log-path', type=str, default=None)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--cpus', type=int, default=5)
    args = parser.parse_args()

    if args.log_path is None:
        args.log_path = './loaded_logs/' + args.env
        if not os.path.isdir(args.log_path):
            os.mkdir(args.log_path)

    env_str = args.env.lower()
    if env_str == 'base':
        from body.PinkPanther.PinkPantherEnv import PinkPantherEnv
        Env = PinkPantherEnv
    elif env_str == 'rand':
        Env = RandPinkPantherEnv
    elif env_str == 'hard':
        Env = HardPinkPantherEnv
    elif env_str == 'easy':
        Env = EasyPinkPantherEnv
    else:
        from body.PinkPanther.pinkpanther import PinkPantherEnv
        env_str = 'base'
        Env = PinkPantherEnv
    try:
        runs = pickle.load(open(args.log_path+'_'+str(args.episodes)+'_loaded_MPC_results.pkl', 'rb'))
        print('Loaded past runs')
    except:
        print('Failed to load runs starting fresh')
        runs = dict()
    Q, P = [], []


    for i in range(min(args.cpus, args.trials)):
        q_in = Queue()
        q_out = Queue()
        p = Process(target=process, args=(q_in, q_out, env_str, args.graph_path, args.data_path, args.episodes))
        p.start()
        q_in.put(i)
        Q.append((q_in, q_out))
        P.append(p)
    start = time.time()
    node_ids = np.concatenate([np.arange(11), np.arange(98)*10+20])
    # node_ids = np.arange(1000)
    test_env, train_envs = get_envs(0, env_str, args.graph_path, args.data_path)
    for i in range(len(node_ids)):
        node_id = node_ids[i]
        if node_id in runs:
            sys.stdout.write('Completed '+str(round(float(100*i)/float(len(node_ids)), 2))+'% in '+str(round(time.time()-start, 3))+'s                                                    \r')
            continue
        run = []
        # for _ in range(args.trials):
        #     test_env, train_envs = get_envs(node_id, env_str, args.graph_path, args.data_path)
        #     params = find_params(train_envs, args.episodes)
        #     score = test_params(params, [test_env])
        #     run.append(score)
        runs[node_id] = parallelize([node_id for _ in range(args.trials)], P, Q)

        pickle.dump(runs, open(args.log_path+'_'+str(args.episodes)+'_loaded_MPC_results.pkl', 'wb+'))
        sys.stdout.write('Completed '+str(round(float(100*i)/float(len(node_ids)), 2))+'% in '+str(round(time.time()-start, 3))+'s                                                    \r')
    print('Done                                                                   ')


