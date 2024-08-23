import numpy as np
import numba as nb

@nb.njit
def game(AM, pRes, r, m, N):
    sumarr = np.zeros_like(pRes)
    player_count = np.zeros_like(pRes)
    net_each_pgg=AM
    sumarr = np.sum(net_each_pgg, axis=1)

    player_count=sumarr.reshape(N,1)

    pres=pRes.reshape(N,1)
    
    PG_accumulated=r*m*net_each_pgg.dot(pres)
    PG_obt=PG_accumulated-m*np.multiply(player_count,pres)

    return(PG_obt)
#-----------------------------------------------------
