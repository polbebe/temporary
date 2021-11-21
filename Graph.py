import numpy as np
from collections import defaultdict
import time

import torch
from train_models import robotTrain, robotPreTrain
from traversal.NavigableGraph import HNSW
from traversal.Greedy import GreedyTraversal
from traversal.BruteForce import BruteForce
from traversal.Random import Random
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
        self.tested = 0

    def load_G(self, path):
        edgeDict = defaultdict(list, json.load(open(path)))
        G = list(edgeDict.keys())
        return edgeDict, G

    def test_dist(self, a, b, dev='cuda'):
        # A is the new one
        # B is the old one
        self.tested += 1
        data_path, model_path = self.data_path, self.model_path
        env_a, env_b = a.split("_")[0], b.split("_")[0]
        robot_a, robot_b = a.split("_")[1], b.split("_")[1]
        data_a = np.load(data_path + env_a + '_train_10000_' + robot_a + '.npy', allow_pickle=True)[-2000:]
        data_b = np.load(data_path + env_b + '_train_10000_' + robot_b + '.npy', allow_pickle=True)[-2000:]
        model_a = torch.load(model_path + env_a + '_model_' + robot_a + '.pt').to(dev)
        model_b = torch.load(model_path + env_b + '_model_' + robot_b + '.pt').to(dev)

        # transform = np.exp
        transform = lambda x: x
        if (a,a) not in self.weights:
            self.weights[(a, a)] = transform(test_valid(model_a, data_a))
        if (a,b) not in self.weights:
            self.weights[(a, b)] = transform(test_valid(model_a, data_b))
        if (b,b) not in self.weights:
            self.weights[(b, b)] = transform(test_valid(model_b, data_b))
        if (b,a) not in self.weights:
            self.weights[(b, a)] = transform(test_valid(model_b, data_a))

        # Edge Weight 1
        # a_norm = self.weights[(b, a)] / self.weights[(a, a)]
        # b_norm = self.weights[(a, b)] / self.weights[(b, b)]
        # edge_wt = np.max([a_norm, b_norm])

        # Edge Weight 2
        # a_norm = self.weights[(b, a)] / self.weights[(a, a)]
        # b_norm = self.weights[(a, b)] / self.weights[(b, b)]
        # edge_wt = np.mean([a_norm, b_norm])

        # Edge Weight 3
        # edge_wt = (self.weights[(a, b)] + self.weights[(b, a)]) / (self.weights[(a, a)] + self.weights[(b, b)])

        # Edge 4
        edge_wt = self.weights[(b, a)]
        return edge_wt

        # env_b = b.split("_")[0]
        # robot_b = b.split("_")[1]
        # model_b = torch.load(self.model_path + env_b + '_model_' + robot_b + '.pt').to(dev)
        #
        # env_a = a.split("_")[0]
        # robot_a = a.split("_")[1]
        # data_a = np.load(self.data_path + env_a + '_train_2000_' + robot_a + '.npy', allow_pickle=True)[:1000]
        #
        # edge_wt = test_valid(model_b, data_a)
        # self.weights[(b,a)] = edge_wt
        # self.weights[(a,b)] = edge_wt
        # return edge_wt

    def test(self, k, algoClass, Net, log, G_path=None, max_batch=6):
        log_file = open(log+'_log.txt', 'w+')
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
            edgeDict[G[0]] = []
        else:
            edgeDict, G = self.load_G(G_path)
        algo = algoClass(G, edgeDict, self.test_dist)

        i = len(G)
        start = time.time()
        while i < len(self.envs) * self.robots:
            t = time.time()
            N, inputs = [], []

            batch_size = max(min(max_batch, int(np.log(i)/np.log(2))), 1)
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
                if (i % 100 == 0):
                    with open(log+".json", "w") as outfile:
                        json.dump(algo.getEdgeDict(), outfile)

            for n in N:
                inputs.append(('train', (n, [])))
            Ls = parallelize(inputs, P, Q)
            # Adding trained node into the graph by connecting to neighbors
            for n in N:
                algo.addNode(n, self.k)
            algo.flush()

            # Collecting Top K neighbors
            inputs = []
            for n in N:
                inputs.append(('train', (n, algo.getNeighbors(n, k))))

            # Training Nodes on their neighbors
            Ls = parallelize(inputs, P, Q)

            print('Node '+str(i)+': '+str(round(time.time()-start,3)))
            print('Dist Calcs: '+str(self.tested))
            print(Ls)
            print('')
            log_file.write('Node '+str(i)+': '+str(round(time.time()-start,3))+'\n')
            log_file.write('Dist Calcs: '+str(self.tested)+'\n')
            log_file.write(str(Ls)+'\n\n')
            log_file.flush()
        print('Process Complete')
        for i in range(len(P)):
            Q[i][0].put(None)
            Q[i][1].get()
        log_file.flush()
        log_file.close()
        return algo

if __name__ == "__main__":
    from models.bnn import BNN
    from models.linear import Linear
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpus', type=int, default=1)  # number of GPUs
    parser.add_argument('--env', type=str, default='rand')
    parser.add_argument('--traversal', type=str, default='greedy')
    parser.add_argument('--robots', type=int, default=1000)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--n_models', type=int, default=1)
    parser.add_argument('--n_elites', type=int, default=1)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--model-path', type=str, default='trained')
    parser.add_argument('--data-path', type=str, default='/data/PinkPanther')
    args = parser.parse_args()

    dataset = args.data_path.split('/')[-1]
    log_file = dataset+'_'+args.env+'_'+args.traversal+'_'+str(args.k)+'_'+str(args.robots)
    print('Started At '+str(time.time()))
    global n_gpus, device
    if args.gpus > 0:
        n_gpus = args.gpus
        device = "cuda"
    else:
        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            device = 'cpu'

    if args.traversal.lower() == 'bruteforce':
        Traversal = BruteForce
    elif args.traversal.lower() == 'random':
        Traversal = Random
    elif args.traversal.lower() == 'greedy':
        Traversal = GreedyTraversal
    elif args.traversal.lower() == 'hnsw':
        Traversal = HNSW
    else:
        print('No Valid traversal chosen defaulting to Greedy')
        Traversal = GreedyTraversal

    from models.seq_ensemble import Ensemble
    pop_size=args.n_models
    n_elites=args.n_elites
    Network = lambda state_dim, act_dim: Ensemble([Linear(state_dim, act_dim) for _ in range(pop_size)], pop_size, n_elites)

    print('Training on '+args.env)
    envs = [args.env]
    model_path = args.model_path+'/'+log_file
    data_path = args.data_path+'/'+args.env
    t = Tester(envs, data_path, model_path, k=args.k, d=10, robots=args.robots)
    G = t.test(args.k, Traversal, Network, log=log_file, G_path=args.load)
    with open(log_file+".json", "w") as outfile:
        json.dump(G.getEdgeDict(), outfile)
