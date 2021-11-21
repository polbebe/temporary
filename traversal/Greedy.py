import numpy as np
import random
import heapq
from utils import parallelize
from collections import defaultdict

class GreedyTraversal:
    def __init__(self, G, edgeDict, dist, P=None, Q=None):
        self.dist = dist
        self.G = G
        self.edgeDict = edgeDict
        self.tmpDict = defaultdict(list)
        self.P = P
        self.Q = Q
        # self.update_weights()

    def connect(self, a, b, d=None):
        if d is None:
            d = self.dist(a, b)
        self.tmpDict[a].append((d, b))
        self.tmpDict[b].append((d, a))
        return d

    def flush(self):
        for n in self.tmpDict.keys():
            if n not in self.edgeDict:
                self.G.append(n)
            self.edgeDict[n].extend(self.tmpDict[n])
        self.tmpDict = defaultdict(list)

    def get_start(self):
        return self.get_uniform_start()
        # return self.get_weighted_start()

    def get_uniform_start(self):
        return self.G[np.random.randint(0, len(self.G))]

    def get_weighted_start(self):
        if len(self.G) > 1:
            n_edges = [1/len(self.edgeDict[key]) for key in self.edgeDict.keys()]
            return random.choices(self.G, weights=n_edges, k=1)[0]
        else:
            return self.G[0]

    def getNeighbors(self, n, k, n_runs=4):
        if n in self.edgeDict:
            return self.edgeDict[n][:k]
        seen = set()
        w = k**3
        top_k = []
        seen.add(n)
        runs = 0
        # Repeat until all nodes visited or we have filled k nodes
        while len(seen) < len(self.G) + 1 and (runs < n_runs or len(top_k) < k):
            temp_k = []

            # Random sample starting point
            u = self.get_start()
            if u in seen:
                continue
            seen.add(u)
            d = self.dist(n, u)
            stack = [(d, u)]

            # Initializing Max Heap temp_k with the first seen element
            temp_k.append((-d, u))
            while stack:
                d, u = stack.pop()
                # print('Node: '+u+' with d='+str(d)+' and '+str(len(self.edgeDict[u]))+' edges')
                # querying all neighbors and continuing if lower than best seen so far
                if self.P is not None:
                    inputs = []
                    # this limits the searchable edges to k^3 which prevents the search from devolving into O(n) in the
                    # worst case scenario like that of a wheel graph
                    p = np.random.permutation(len(self.edgeDict[u]))
                    for i in range(min(len(p), w)):
                        _, v = self.edgeDict[u][p[i]]
                    # for _, v in self.edgeDict[u]:
                        if v in seen: continue
                        seen.add(v)
                        inputs.append(('dist', (n, v)))
                    D = parallelize(inputs, self.P, self.Q)
                else:
                    D = []
                    p = np.random.permutation(len(self.edgeDict[u]))
                    for i in range(min(len(p), w)):
                        _, v = self.edgeDict[u][p[i]]
                        if v in seen: continue
                        seen.add(v)
                        new_d = self.dist(n, v)
                        D.append((n, v, new_d))

                for _, v, new_d in D:
                    # Only if the new_d is better than the worst element in the heap
                    if len(temp_k) < k or new_d < -temp_k[0][0]:
                        heapq.heappush(temp_k, (-new_d, v))
                        if len(temp_k) > k:
                            heapq.heappop(temp_k)
                    if new_d < d:
                        stack = [(new_d, v)]
                        d = new_d

            # Merging top_k so far with top_k this iteration (temp_k)
            new_k = []
            while temp_k:
                new_k.append(heapq.heappop(temp_k))
            merged = heapq.merge(top_k, new_k)
            top_k = [m for m in merged][-k:]
            runs += 1

        assert len(seen) == len(self.G) + 1 or len(top_k) == k
        # print(top_k)
        # print(float(len(seen)))
        return top_k

    def addNode(self, n, k, n_runs=4):
        neighbors = self.getNeighbors(n, k, n_runs)
        # Collecting the top_k and connecting the test node to those verticies
        w, e = [], []
        for d, u in neighbors:
            self.connect(n, u, -d)
            w.append(-d)
            e.append((n, u))

        # Add the newly connected node to the graph
        return e, w

    def getEdgeDict(self):
        return self.edgeDict