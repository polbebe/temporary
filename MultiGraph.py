import numpy as np
from collections import defaultdict
import time

import torch
from train_models import robotTrain, robotPreTrain
from traversal.NavigableGraph import HNSW
from traversal.Greedy import GreedyTraversal
from utils import parallelize, test_valid

import json
from torch.multiprocessing import Process, Queue
import argparse

# Multiprocessing function to be called for training
def train_process(q_in, q_out, Net, dist, data_path, model_path):
    global n_gpus
    id = q_in.get()
    if n_gpus > 0:
        torch.cuda.set_device(id%n_gpus)
        device = 'cuda'
    else:
        device = 'cpu'
    print('Process '+str(id)+' started.')
    while True:
        got = q_in.get()
        if got is None: break
        cmd, obj = got
        if cmd == 'train':
            n, neighbors = obj
            L = robotPreTrain(n, neighbors, Net, data_path, model_path)
            q_out.put(L)
            del n, neighbors, obj
        elif cmd == 'dist':
            a, b = obj
            d = dist(a, b, device)
            q_out.put((a, b, d))
            del a, b, d, obj
    q_out.put(None)
    print('Process '+str(id)+' ended.')

class Tester:
    def __init__(self, envs, data_path, model_path, k=4, d=10, robots=100000):
        self.k = k
        self.d = d
        self.envs = envs
        self.robots = robots
        self.data_path = data_path
        if self.data_path[-1] != '/': self.data_path += '/'
        self.model_path = model_path
        if self.model_path[-1] != '/': self.model_path += '/'
        self.weights = {}

    def load_G(self, path):
        edgeDict = defaultdict(list, json.load(open(path)))
        G = list(edgeDict.keys())
        return edgeDict, G

    def test_dist(self, b, a, dev='cuda'):
        if((b,a) in self.weights):
            return self.weights[(b,a)]
        env_b = b.split("_")[0]
        robot_b = b.split("_")[1]
        model_b = torch.load(self.model_path + env_b + '_model_' + robot_b + '.pt').to(dev)

        env_a = a.split("_")[0]
        robot_a = a.split("_")[1]
        data_a = np.load(self.data_path + env_a + '_train_1000_' + robot_a + '.npy', allow_pickle=True)

        edge_wt = test_valid(model_b, data_a)
        self.weights[(b,a)] = edge_wt
        self.weights[(a,b)] = edge_wt
        return edge_wt

    def test(self, k, algoClass, Net, n_runs, env, batch=False, G_path=None):
        if batch:
            return self.test_batch(k, algoClass, Net, n_runs, env, G_path)
        else:
            return self.test_seq(k, algoClass, Net, n_runs, env, G_path)

    def test_seq(self, k, algoClass, Net, n_runs, env, G_path=None):
        nodes = []
        for i in range(len(envs) * self.robots):
            nodes.append(i)

        if G_path is None:
            # Inserting one node into graph manually
            G = [envs[0]+'_'+str(0)]
            robotTrain(envs[0], 0, Net, self.data_path, self.model_path)
            edgeDict = defaultdict(list)
        else:
            edgeDict, G = self.load_G(G_path)

        algo = algoClass(G, edgeDict, self.test_dist)

        start = time.time()
        for i in range(len(G), len(self.envs * self.robots)):
            t = time.time()
            if (i % 500 == 0):
                with open(env+"_1.json", "w") as outfile:
                    json.dump(algo.getEdgeDict(), outfile)
            env = envs[i // self.robots]
            robot = i % self.robots

            # Defining a node by its relation to an env and a robot ID
            n = str(env) + "_" + str(robot)
            # Collecting Top K neighbors
            neighbors = algo.getNeighbors(n, k, n_runs)
            neighbor_t = time.time()
            # Training Node on these neighbors
            robotPreTrain(n, neighbors, Net, self.data_path, self.model_path)
            pretrain_t = time.time()
            # Adding trained node into the graph by connecting to neighbors
            algo.addNode(n, neighbors)
            add_t = time.time()
            print('Node '+str(i)+' :'+str(round(time.time()-start,3)))
            print('\tNeighbors Time: '+str(round(neighbor_t - t,3)))
            print('\tPretrain Time: '+str(round(pretrain_t - neighbor_t,3)))
        return algo

    def test_batch(self, k, algoClass, Net, n_runs, env, G_path=None, max_batch=6):
        Q, P = [], []
        for i in range(max_batch):
            q_in = Queue()
            q_out = Queue()
            p = Process(target=train_process, args=(q_in, q_out, Net, self.test_dist, self.data_path, self.model_path))
            p.start()
            q_in.put(i)
            Q.append((q_in, q_out))
            P.append(p)

        nodes = []
        for i in range(len(envs) * self.robots):
            nodes.append(i)

        if G_path is None:
            # Inserting one node into graph manually
            G = [envs[0]+'_'+str(0)]
            robotTrain(envs[0], 0, Net, self.data_path, self.model_path)
            edgeDict = defaultdict(list)
        else:
            edgeDict, G = self.load_G(G_path)
        algo = algoClass(G, edgeDict, self.test_dist)

        i = len(G)
        start = time.time()
        while i < len(self.envs) * self.robots:
            t = time.time()
            N, neighbors = [], []

            batch_size = max(min(max_batch, int(np.log(len(G))/np.log(2))), 1)
            # Collecting nodes
            for j in range(batch_size):
                if i >= len(self.envs) * self.robots:
                    break
                env = envs[i // self.robots]
                robot = i % self.robots
                i += 1
                # Defining a node by its relation to an env and a robot ID
                n = str(env) + "_" + str(robot)
                N.append(n)
                if (i % 500 == 0):
                    with open(env+"_1.json", "w") as outfile:
                        json.dump(algo.getEdgeDict(), outfile)

            # Adding trained node into the graph by connecting to neighbors
            for j in range(len(N)):
                algo.addNode(N[j],self.k)

            # Collecting Top K neighbors
            for n in N:
                neighbors.append(algo.getNeighbors(n, k, n_runs))
            # Training Nodes on their neighbors
            inputs = [('train', (N[j], neighbors[j])) for j in range(len(N))]
            Ls = parallelize(inputs, P, Q)
            print(Ls)
            print('Node '+str(len(G))+' :'+str(round(time.time()-start,3)))
        print('Process Complete')
        for i in range(len(P)):
            Q[i][0].put(None)
            Q[i][1].get()
        return algo

if __name__ == "__main__":
    from models.bnn import BNN
    # from models.linear import Linear
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpus', type=int, default=1)  # number of GPUs
    parser.add_argument('--env', type=str, default='rand')
    parser.add_argument('--robots', type=int, default=10000)
    parser.add_argument('--runs', type=int, default=4)
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--n_models', type=int, default=5)
    parser.add_argument('--n_elites', type=int, default=5)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--model-path', type=str, default='trained')
    parser.add_argument('--data-path', type=str, default='data/PinkPanther')
    args = parser.parse_args()
    print('Started At '+str(time.time()))
    global n_gpus, device
    if args.gpus > 0:
        n_gpus = args.gpus
        device = "cuda"
    else:
        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            device = 'cpu'

    from models.seq_ensemble import Ensemble
    pop_size=args.n_models
    n_elites=args.n_elites
    Network = lambda state_dim, act_dim: Ensemble([BNN(state_dim, act_dim) for _ in range(pop_size)], pop_size, n_elites)

    print('Training on '+args.env)
    envs = [args.env]
    model_path = args.model_path+'/'+args.env
    data_path = args.data_path+'/'+args.env
    t = Tester(envs, data_path, model_path, k=args.k, d=10, robots=args.robots)
    G = t.test(args.k, HNSW, Network, args.runs, args.env, G_path=args.load, batch=True)
    with open(args.env+"_1.json", "w") as outfile:
        json.dump(G.edgeDict, outfile)
