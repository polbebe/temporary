import numpy as np
from heapq import heapify, heappush, heappop
from collections import defaultdict
import math

class HNSW:
    def __init__(self, G, edgeDict, dist):
        #zeroNode = (0,0,0,0,0,0,0,0,0,0)
        #firstNode = tuple(G[0][0])

        self.dist = dist
        #self.G = {0:{zeroNode:{firstNode:self.dist(firstNode,zeroNode)},firstNode:{zeroNode:self.dist(firstNode,zeroNode)}}}  # key = layer index and value = induced subgraph
        self.G = {0:{G[0]:{}}}
        self.entryPoint = [(0,G[0])]
        self.M = 10
        self.efConsruction = 100
        self.Max0 = 2*self.M
        self.mu = 1/np.log(self.M)
        self.edgeDict = edgeDict

    def hash(self, n):
        return n.tobytes()

    def SearchLayerHNSW(self, q, ep, ef, lc):
        v = set()
        C = []
        W = []
        for i in range(len(ep)):
            W.append((-1*self.dist(ep[i][1],q),ep[i][1]))
            C.append((self.dist(ep[i][1],q),ep[i][1]))
            v.add(ep[i][1])

        heapify(W)

        heapify(C)

        while(len(C)>0):
            c = heappop(C)
            f = W[0]
            #print(c)
            #print(q)
            #print(f)
            if(self.dist(c[1],q)>self.dist(f[1],q)):
                break

            try:
                if(len(c[1])>0):
                    pass
            except:
                print(c[1])

            neighbours = self.G[lc][c[1]]
            #print(neighbours)
            for key,val in neighbours.items():
                if(key not in v):
                    v.add(key)
                    f = W[0]
                    if((self.dist(key,q) < self.dist(f[1],q)) or (len(W) < ef)):
                        heappush(C,(self.dist(key,q),key))
                        heappush(W,(-1*self.dist(key,q),key))

                        if(len(W)>ef):
                            heappop(W)

        W_updated = []
        for i in range(len(W)):
            W_updated.append((self.dist(W[i][1],q),W[i][1]))

        return W_updated




    def SelectNeighbors(self, q, C, M):
        ng = []
        for i in range(len(C)):
            ng.append((self.dist(C[i][1],q),C[i][1]))
        heapify(ng)

        return ng[:M]

    def InsertHNSW(self, G, q, M, Mmax, efCon, mu):
        W = [] #list of candidates. Each element is a tuple (wt, node). It is a min heap
        ep = self.entryPoint # entry point in the multi-graph
        L = len(self.G) -1 #index of the top layer
        l = math.floor(math.log(np.random.uniform(low = 0, high =1))*(-1)*mu)
        lc = L

        while(lc>=l+1):
            W = self.SearchLayerHNSW(q,ep,1,lc)
            print("Found the first neighbour")
            ep = W[0] #Fetching the optimal tuple
            lc = lc -1
        lc = min(L,l)
        while(lc>=0):
            W = self.SearchLayerHNSW(q,ep,efCon,lc)
            #print("W is ")
            #print(W)
            neighbours = self.SelectNeighbors(q,W,M)

            if(len(self.G[lc])==0):
                self.G[lc] = {q:{}}
            elif(q not in self.G[lc]):
                self.G[lc][q] = {}

            #adding bidirectional edges from neighbours to q at level l
            for i in range(len(neighbours)):
                """
                print("Graph is ")
                print(self.G[lc])
                print("Node that is getting inserted is")
                print(q)
                print("Neighbour under consideration")
                print(neighbours[i][1])
                """
                self.G[lc][q][neighbours[i][1]] = neighbours[i][0]
                self.G[lc][neighbours[i][1]][q] = neighbours[i][0]

            for i in range(len(neighbours)):
                eConn = self.G[lc][neighbours[i][1]] #This is a dictionary containing node : wt
                eConnList = []
                for key,val in eConn.items():
                    eConnList.append((val,key))
                    
                
                if((len(eConn)>Mmax and lc!=0) or (lc==0 and len(eConn)>self.Max0)):

                    eNewConn = self.SelectNeighbors(neighbours[i][1],eConnList,Mmax)
                    self.G[lc][neighbours[i][1]] = {}

                    for j in range(len(eNewConn)):
                        self.G[lc][neighbours[i][1]][eNewConn[j][1]] = eNewConn[j][0]
                    for key,val in eConn.items():
                        if(key not in self.G[lc][neighbours[i][1]]):
                            self.G[lc][key].pop(neighbours[i][1])
                

            lc = lc-1
            ep = W
        if(l>L):
            self.entryPoint = [(0,q)]



    def KNNSearch(self, G, q, K, ef):
        pass

    def addNode(self, n, k):
        #n = tuple(n[0])
        self.InsertHNSW(self.G,n,self.M,12,self.efConsruction,self.mu)
        neighbours = self.G[0][n]
        w = []
        e = []
        for key,val in neighbours.items():
            w.append(val)
            e.append((n,key))
        #print("Inserted node"+str(n))
        return e,w


    def getNeighbors(self, n, k, n_runs):
        ng = []
        for key,val in self.G[0][n].items():
            ng.append((val,key))
        heapify(ng)

        return ng[:k]
    
    def getEdgeDict(self):
        edges = {}
        
        for k, val in self.G[0].items():
            edges[k] = []
            for p,q in val.items():
                edges[k].append((q,p))
                
        self.edgeDict = edges