
import numpy as np

import numba
#from numba import jitclass, int64
#-------------------------------------------------
#game : 1 round of PGG over the entire network for all nodes
#-------------------------------------------------
@numba.njit
def game(AM, pRes, r, m, N):#pRes:Player Response AM:Adjacency Matrix
    sumarr = np.zeros_like(pRes)
    player_count = np.zeros_like(pRes)
    #network on which the PGG will be played
    net_each_pgg=AM
    sumarr = np.sum(net_each_pgg, axis=1)#.reshape(N,1)

    #number of players participating in 'i'th PGG is player_count
    player_count=sumarr.reshape(N,1)

    pres=pRes.reshape(N,1)
    
    #PGG: Total PG for the game around 'i'th node is stored.
    #C=1 D=0, thus dot product works.
    PG_accumulated=r*m*net_each_pgg.dot(pres)
    PG_obt=PG_accumulated-m*np.multiply(player_count,pres)

    return(PG_obt)
#-----------------------------------------------------
