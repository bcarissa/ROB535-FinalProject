
import numpy as np
from functools import partial
from numpy.random import default_rng
# rng = default_rng()
rng = default_rng(seed=3)  #set seed to get deterministic random number generator

from copy import deepcopy, copy

class myStruct:
    pass

init = myStruct()

def Gfun(mu,u):
    output = np.array([[1,0,(u[0]*np.cos(mu[2]+u[1]))/u[1]-(u[0]*np.cos(mu[2]))/u[1]],\
                        [0,1,(u[0]*np.sin(mu[2]+u[1]))/u[1]-(u[0]*np.sin(mu[2]))/u[1]],\
                        [0,0,1]])
    return output

def Vfun(mu,u):
    output = np.array([[np.sin(mu[2]+u[1])/u[1]-np.sin(mu[2])/u[1], (u[0]*np.cos(mu[2]+u[1]))/u[1]-(u[0]*np.sin(mu[2]+u[1]))/(u[1]**2)+(u[0]*np.sin(mu[2]))/(u[1]**2), 0],\
                       [np.cos(mu[2])/u[1]-np.cos(mu[2]+u[1])/u[1], (u[0]*np.cos(mu[2]+u[1]))/(u[1]**2)+(u[0]*np.sin(mu[2]+u[1]))/u[1]-(u[0]*np.cos(mu[2]))/(u[1]**2), 0],\
                       [0,                                          1,                                                                                             1]])

    return output

def Hfun(landmark_x, landmark_y, mu_pred, z_hat):
    output = np.array([
        [(landmark_y-mu_pred[1])/(z_hat[1]**2),   -(landmark_x-mu_pred[0])/(z_hat[1]**2),-1],\
        [-(landmark_x-mu_pred[0])/z_hat[1],       -(landmark_y-mu_pred[1])/z_hat[1],0]])
    return output
    


def filter_initialization(sys, initialStateMean, initialStateCov, filter_name):
    init.mu = initialStateMean
    init.Sigma = initialStateCov
    from filter.DummyFilter import DummyFilter
    filter = DummyFilter(sys, init)
            
    return filter