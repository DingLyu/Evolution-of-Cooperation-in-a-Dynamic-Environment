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
import rewire
import initialize as init
import windowing

sns.set()

tTime = time.time()
totTime = 0.

totNP = 100
trials = 10
rounds = 750

eps = 0.02
c = 1.
r = 2.
rmin = 1
rmax = 3
b = r*c
ini_re = 0.3


def adjust_env_factor(x, pcmean):
    k1 = 0.25
    theta = 1
    delta_x = k1 * x * (1 - x) * ((1 + theta) * pcmean - 1)
    x_ = x + delta_x
    return np.clip(x_, 0, 1)


h_arr = np.arange(0.025, 0.99, 0.025)
h_arr = np.round(h_arr, 5)
beta_arr = np.logspace(-2.2, 0, 40)/c
print(h_arr, beta_arr)
beta_arr = beta_arr[::-1]

h1_arr = h_arr[::2]
b_arr = beta_arr[::2]
b1_arr = []
for b in b_arr:
    b1_arr.append(np.format_float_scientific(b, trim='-'))

eqbTime = 250

print(len(beta_arr), len(h_arr))

coopfrac_arr = np.zeros((len(beta_arr), len(h_arr), trials, rounds))
sat_arr = np.zeros((len(beta_arr), len(h_arr), trials, rounds))
asp_arr = np.zeros((len(beta_arr), len(h_arr), trials, rounds))
Csat_arr = np.zeros((len(beta_arr), len(h_arr), trials, rounds))
Dsat_arr = np.zeros((len(beta_arr), len(h_arr), trials, rounds))
Casp_arr = np.zeros((len(beta_arr), len(h_arr), trials, rounds))
Dasp_arr = np.zeros((len(beta_arr), len(h_arr), trials, rounds))
Env_arr = np.zeros((len(beta_arr), len(h_arr), trials, rounds))
hInd = -1
for hab in h_arr:
    hTime = time.time()
    print('start hab=', hab)
    hInd = hInd + 1
    betaInd = -1
    for bet in beta_arr:
        print('start beta=', bet)
        betaInd = betaInd + 1
        for it in range(trials):
            AdjMat = init.init_adjmat(totNP, ini_re)
            x = 0.1
            r = rmin * (1 - x) + rmax * x
            [aspA, satA, pcA, payA, cpayA, habA, betaA, actA] = init.init_arr(totNP, AdjMat, bet, hab, r, c)
            for i_main in range(rounds):

                r = rmin * (1 - x) + rmax * x
                pay = PGG.game(AdjMat, actA, r, c, totNP)
                [coopfrac, sais, asp, Csat, Dsat, Casp, Dasp] = Record.measure(actA, satA, aspA, AdjMat)
                x = adjust_env_factor(x, coopfrac)

                coopfrac_arr[betaInd, hInd, it, i_main] = coopfrac
                sat_arr[betaInd, hInd, it, i_main] = sais
                asp_arr[betaInd, hInd, it, i_main] = asp
                Csat_arr[betaInd, hInd, it, i_main] = Csat
                Dsat_arr[betaInd, hInd, it, i_main] = Dsat
                Casp_arr[betaInd, hInd, it, i_main] = Casp
                Dasp_arr[betaInd, hInd, it, i_main] = Dasp
                Env_arr[betaInd, hInd, it, i_main] = x

                [aspA, satA, pcA, actA, cpayA] = Record.update_iterated(pay, aspA, satA, payA, cpayA, pcA, habA, betaA, actA, eps, totNP)


    print('h='+str(hab)+' required: ', (time.time()-hTime))
    print('total time taken: ', (time.time()-tTime))
print(time.time()-tTime)
trounds = np.arange(0, rounds, 1)

np.savez("data/Fig1_x0_01.npz", beta=beta_arr, h=h_arr,
         coopfrac=coopfrac_arr, sat=sat_arr,
         Csat=Csat_arr, Dsat=Dsat_arr, Env=Env_arr)
