import numpy as np
import random
import heapq
from utils import parallelize
from collections import defaultdict

class Random:
    def __init__(self, G, edgeDict, dist, P=None, Q=None):
        self.dist = dist
        self.G = G
        self.edgeDict = edgeDict
        self.tmpDict = defaultdict(list)
        self.P = P
        self.Q = Q

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

    def getNeighbors(self, n, k):
        if n in self.edgeDict:
            return self.edgeDict[n][:k]
        p = np.random.permutation(len(self.G))[:k]
        top_k = [(self.dist(n, self.G[i]), self.G[i]) for i in p ]
        return top_k

    def addNode(self, n, k):
        neighbors = self.getNeighbors(n, k)
        # Collecting the top_k and connecting the test node to those verticies
        w, e = [], []
        for d, u in neighbors:
            self.connect(n, u, d)
            w.append(d)
            e.append((n, u))

        # Add the newly connected node to the graph
        return e, w

    def getEdgeDict(self):
        return self.edgeDict