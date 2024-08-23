import numpy as np
import numba as nb
import networkx as nx
import scipy as sc

@nb.njit
def init_arr(N, AM, bet, hab, r, c, pc = 0.5):
    satisfaction = np.zeros((N, 1))
    aspiration = np.zeros((N, 1))
    pC = np.zeros((N, 1))
    payoff = np.zeros((N, 1))
    cumpayoff = np.zeros((N, 1))
    habituation = np.zeros((N, 1))
    beta = np.zeros((N, 1))
    action = np.zeros((N, 1))
    choose = np.array([0., 1.])
    
    for ind in range(N):
        satisfaction[ind] = 0.
        deg = len(np.where(AM[ind]==1)[0])
        aspiration[ind] = deg*c*(r-1)/2.
        beta[ind] = bet
        habituation[ind] = hab
        payoff[ind] = 0.
        pC[ind] = 0.5
        #action[ind] = nb.float64(np.random.random() < 0.5)
        action[ind] = np.random.choice(choose)
        cumpayoff[ind] = 0.
    return(aspiration, satisfaction, pC,
           payoff, cumpayoff, habituation, beta, action)


@nb.njit
def init_adjmat(N, p):
    adjacency_matrix = np.zeros((N, N))
    for i in range(N):
        for j in np.arange(i+1, N, 1):
            response = nb.float64(np.random.random() < p)
            adjacency_matrix[i][j] = response
            adjacency_matrix[j][i] = response
    return (adjacency_matrix)

@nb.jit
def ba_network(n, m=16):
    adjacency_matrix = np.zeros((n, n), dtype=np.float64)

    for i in range(m):
        for j in range(i + 1, m):
            adjacency_matrix[i, j] = 1
            adjacency_matrix[j, i] = 1

    degrees = np.zeros(n, dtype=np.float64)
    degrees[:m] = m - 1

    for i in range(m, n):
        prob_dist = degrees[:i] / np.sum(degrees[:i])
        targets = np.random.choice(np.arange(i), size=m, replace=False, p=prob_dist)

        for target in targets:
            adjacency_matrix[i, target] = 1
            adjacency_matrix[target, i] = 1

        degrees[i] = m
        degrees[targets] += 1

    return adjacency_matrix

@nb.jit(forceobj=True)
def price_network(n, gamma):
    adjacency_matrix = np.zeros((n, n), dtype=np.float64)
    p = 1 / (gamma - 1)
    m0 = 16
    Array = [0, 0, 1, 0, 2]
    for i in range(m0):
        for j in range(i + 1, m0):
            adjacency_matrix[i, j] = 1
            adjacency_matrix[j, i] = 1
    m = 15
    for i in range(m0, n):
        temp = []
        while len(temp) < m:
            if nb.float64(np.random.random() < p):
                t = np.random.choice(Array)
            else:
                t = np.random.randint(0, i)
            if t in temp:
                continue
            else:
                temp.append(t)
        for j in temp:
            adjacency_matrix[i][j] = 1
            adjacency_matrix[j][i] = 1
        Array.extend(temp)
    return adjacency_matrix


