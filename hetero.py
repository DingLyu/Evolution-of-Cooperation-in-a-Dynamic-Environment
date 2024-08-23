import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import scipy.sparse as scsp
import networkx as nx
import pandas as pd
import time
from datetime import datetime
import numba as nb
from numba import int64, float64
import seaborn as sns
import PGG
import Record_vectorized as Record

import initialize as init


sns.set()

tTime = time.time()
totTime = 0.

totNP = 1000
trials = 100
rounds = 750

eps = 0.02
c = 1.
r = 2.
rmin = 1
rmax = 3
b = r * c
ini_re = 0.3


def adjust_env_factor(x, pcmean, theta):
    k1 = 0.25
    delta_x = k1 * x * (1 - x) * ((1 + theta) * pcmean - 1)
    x_ = x + delta_x
    return np.clip(x_, 0, 1)

# x0_arr = np.arange(0.1, 1.0, 0.1)
# x0_arr = x0_arr[::-1]
x0_arr =np.array([0.5, 0.7])
# theta_arr = np.arange(0.1, 2.1, 0.1)
# gamma_arr = np.array([2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 5, 10])
gamma_arr = np.array([2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4])
hab = 0.4
bet = 1


eqbTime = 250



coopfrac_arr = np.zeros((len(x0_arr), len(gamma_arr), trials, rounds))
sat_arr = np.zeros((len(x0_arr), len(gamma_arr), trials, rounds))
asp_arr = np.zeros((len(x0_arr), len(gamma_arr), trials, rounds))
Csat_arr = np.zeros((len(x0_arr), len(gamma_arr), trials, rounds))
Dsat_arr = np.zeros((len(x0_arr), len(gamma_arr), trials, rounds))
Casp_arr = np.zeros((len(x0_arr), len(gamma_arr), trials, rounds))
Dasp_arr = np.zeros((len(x0_arr), len(gamma_arr), trials, rounds))
Env_arr = np.zeros((len(x0_arr), len(gamma_arr), trials, rounds))


hInd = -1
hTime = time.time()
for gamma in gamma_arr:
    hInd = hInd + 1
    betaInd = -1
    for x0 in x0_arr:
        betaInd = betaInd + 1
        for it in range(trials):
            print(gamma, it, time.time()-hTime)
            AdjMat = init.price_network(totNP, gamma)
            x = x0
            r = rmin * (1 - x) + rmax * x
            [aspA, satA, pcA, payA, cpayA, habA, betaA, actA] = init.init_arr(totNP, AdjMat, bet, hab, r, c)

            for i_main in range(rounds):
                r = rmin * (1 - x) + rmax * x
                pay = PGG.game(AdjMat, actA, r, c, totNP)

                [coopfrac, sais, asp, Csat, Dsat, Casp, Dasp] = Record.measure(actA, satA, aspA, AdjMat)

                x = adjust_env_factor(x, coopfrac, 1)
                coopfrac_arr[betaInd, hInd, it, i_main] = coopfrac
                sat_arr[betaInd, hInd, it, i_main] = sais
                asp_arr[betaInd, hInd, it, i_main] = asp
                Csat_arr[betaInd, hInd, it, i_main] = Csat
                Dsat_arr[betaInd, hInd, it, i_main] = Dsat
                Casp_arr[betaInd, hInd, it, i_main] = Casp
                Dasp_arr[betaInd, hInd, it, i_main] = Dasp
                Env_arr[betaInd, hInd, it, i_main] = x

                # actP = actA.copy()

                [aspA, satA, pcA, actA, cpayA] = Record.update_iterated(pay, aspA, satA, payA, cpayA, pcA, habA, betaA, actA, eps, totNP)

    print('h=' + str(hab) + ' required: ', (time.time() - hTime))
    print('total time taken : ', (time.time() - tTime))

print(time.time() - tTime)
trounds = np.arange(0, rounds, 1)



np.savez("data/specific_2to4.npz", x0=x0_arr, gamma=gamma_arr,
         coopfrac=coopfrac_arr, sat=sat_arr,
         Csat=Csat_arr, Dsat=Dsat_arr, Env=Env_arr)
