import numpy as np
import scipy 
from matplotlib import pyplot as plt
import os
from scipy.linalg import norm as l2norm
import math
import random
import time
def clipper(gd, C):
    # do gradient clipping
    norm_size = l2norm(gd)
    if norm_size>C:
        return C*gd/norm_size
    else:
        return gd
def sigmoid(x):

    return 1/(1+np.exp(-x))

def gaussian_privacy_cost(epsilon, c=1, d=1, delta = 0.01):
    # calculate the privacy cost for Gaussian mechanism
    # c is the bound of |x|, d is the dimension of x
    sens = 2 * c * np.sqrt(d)
    print("sensitivity: ", sens)
    a = np.sqrt(2*np.log(1.25/delta))
    sigma = (a * sens)/epsilon
    return sigma


def data_iter(batch_size, *arrays):
    num_examples = len(arrays[0])
    indices = list(range(num_examples))
    random.shuffle(indices)
    
    for i in range(0, num_examples, batch_size):
        batch_indices = np.array(indices[i: min(i + batch_size, num_examples)])
        yield tuple(array[batch_indices] for array in arrays)


def rr_mechansim(data, alpha):
    '''
    data: indicate function of original data
    discrete differential privacy guarantee mechanism for binary data
    '''
    n_data = len(data)
    p = sigmoid(alpha)
    random_response = np.random.binomial(1, p, n_data)
    rr_vr = np.where(data, random_response, 1-random_response)
    corrected_res = (rr_vr-(1-p))/(2*p-1)
    return rr_vr

def plot_figure(aggregator, global_theta):
    trace_data = np.vstack(aggregator.theta_bar)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    ax[0].plot(trace_data[:,0][100:], 'b', linestyle='--', label='Param1')
    ax[0].plot(trace_data[:,1][100:], 'orange', linestyle='--', label='Param2')
    ax[0].plot(trace_data[:,2][100:], 'g', linestyle='--', label='Param3')
    ax[0].plot(trace_data[:,3][100:], 'r', linestyle='--', label='Param4')
    ax[0].axhline(y=global_theta[0], color='b', linestyle='--')
    ax[0].axhline(y=global_theta[1], color='orange', linestyle='--')
    ax[0].axhline(y=global_theta[2], color='g', linestyle='--')
    ax[0].axhline(y=global_theta[3], color='r', linestyle='--')
    ax[0].set_xlabel('Communication rounds')
    ax[0].set_ylabel('Parameter value')
    ax[0].grid()
    ax[0].legend()
    ym_bar = np.vstack(aggregator.ym)
    ax[1].plot((ym_bar[:,0] - global_theta[0])**2, 'b', linestyle='--', label='Param1')
    ax[1].plot((ym_bar[:,1] - global_theta[1])**2, 'orange', linestyle='--', label='Param2')
    ax[1].plot((ym_bar[:,2] - global_theta[2])**2, 'r', linestyle='--', label='Param3')
    ax[1].plot((ym_bar[:,3] - global_theta[3])**2, 'g', linestyle='--', label='Param4')
    ax[1].legend()
    ax[1].set_xlabel('Communication rounds')
    ax[1].set_ylabel(r'$\| \theta^*-\hat{\theta}\|^2$')
    plt.suptitle('Local_SGD iteration trace')
    plt.tight_layout()
    ax[1].grid()
    plt.show()

def get_index_CI(ym, index, schedule = 5):

    rounds, dim = ym.shape
    Em = np.ones(rounds) * schedule
    sm = np.zeros(rounds)
    qm = np.zeros(rounds)
    Am = np.zeros((rounds, dim,dim))
    bm = np.zeros((rounds, dim))
    Vm = []
    if isinstance(index, int):
        for k in range(0, rounds):
            if k==0:
                sm[k] = 1/Em[k]
                qm[k] = (k+1)**2/Em[k]
                Am[k] = (((k+1)**2)/schedule) * np.outer(ym[k], ym[k])
                bm[k] = (((k+1)**2)/schedule) * ym[k]
            else:
                sm[k] = sm[k-1] + 1/Em[k]
                qm[k] = qm[k-1] + ((k+1)**2)/Em[k]
                Am[k] = Am[k-1] + (((k+1)**2)/schedule) * np.outer(ym[k], ym[k])
                bm[k] = bm[k-1] + (((k+1)**2)/schedule) * ym[k]
            
            if (k+1) % index == 0 :
                const = ((k+1)**2)*sm[k]
                const = 1/const
                temp  = const * (Am[k] - np.outer(ym[k], bm[k])
                                            - np.outer(bm[k], ym[k]) + qm[k]*np.outer(ym[k], ym[k]))
                Vm.append(np.sqrt(np.diag(temp)))
    else:
        for k in range(0, rounds):
            if k==0:
                sm[k] = 1/Em[k]
                qm[k] = (k+1)**2/Em[k]
                Am[k] = (((k+1)**2)/schedule) * np.outer(ym[k], ym[k])
                bm[k] = (((k+1)**2)/schedule)* ym[k]
            else:
                sm[k] = sm[k-1] + 1/Em[k]
                qm[k] = qm[k-1] + (k+1)**2/Em[k]
                Am[k] = Am[k-1] + (((k+1)**2)/schedule) * np.outer(ym[k], ym[k])
                bm[k] = bm[k-1] + (((k+1)**2)/schedule)*ym[k]
            if k in index:
                const = (k+1)**2*sm[k]
                const = 1/const
                temp  = const * (Am[k] - np.outer(ym[k], bm[k])
                                            - np.outer(bm[k], ym[k]) + qm[k]*np.outer(ym[k], ym[k]))
                Vm.append(np.sqrt(np.diag(temp)))
    Vm = np.array(Vm)
    return Vm

def get_CI(ym,schedule):
    rounds, dim= ym.shape
    Em = np.ones(rounds) * schedule
    sm = np.zeros(rounds)
    qm = np.zeros(rounds )
    Am = np.zeros((rounds,dim,dim))
    bm = np.zeros((rounds,dim))
    Vm = np.zeros((rounds,dim))
    for k in range(0, rounds):
        if k==0:
            sm[k] = 1/Em[k]
            qm[k] = (k+1)**2/Em[k]
            Am[k-1] = (((k+1)**2)/schedule) * np.outer(ym[k], ym[k])
            bm[k-1] = (((k+1)**2)/schedule)* ym[k]
        else:
            sm[k] = sm[k-1] + 1/Em[k]
            qm[k] = qm[k-1] + (k+1)**2/Em[k]
            Am[k] = Am[k-1] + (((k+1)**2)/schedule) * np.outer(ym[k], ym[k])
            bm[k] = bm[k-1] + (((k+1)**2)/schedule)*ym[k]
        const = (k+1)**2*sm[k]
        const = 1/const
        temp  = const * (Am[k] - np.outer(ym[k], bm[k])
                                    - np.outer(bm[k], ym[k]) + qm[k]*np.outer(ym[k], ym[k]))
        Vm[k] = np.sqrt(np.diag(temp))
    return Vm


def get_index_CI2(ym, index, schedule = 5, rate =0.05):
    rounds, dim = ym.shape
    Em = np.ones(rounds) * schedule
    Em[:round(rounds*rate)] = 1
    sm = np.zeros(rounds)
    qm = np.zeros(rounds)
    Am = np.zeros((rounds, dim, dim))
    bm = np.zeros((rounds, dim))
    Vm = []
    if isinstance(index, int):
        for k in range(0, rounds):
            if k==0:
                sm[k] = 1/Em[k]
                qm[k] = (k+1)**2/Em[k]
                Am[k] = (((k+1)**2)/Em[k]) * np.outer(ym[k], ym[k])
                bm[k] = (((k+1)**2)/Em[k]) * ym[k]
            else:
                sm[k] = sm[k-1] + 1/Em[k]
                qm[k] = qm[k-1] + ((k+1)**2)/Em[k]
                Am[k] = Am[k-1] + (((k+1)**2)/Em[k]) * np.outer(ym[k], ym[k])
                bm[k] = bm[k-1] + (((k+1)**2)/Em[k]) * ym[k]
            
            if (k+1) % index == 0 :
                const = ((k+1)**2)*sm[k]
                const = 1/const
                temp  = const * (Am[k] - np.outer(ym[k], bm[k])
                                            - np.outer(bm[k], ym[k]) + qm[k]*np.outer(ym[k], ym[k]))
                Vm.append(np.sqrt(np.diag(temp)))
    else:
        for k in range(0, rounds):
            if k==0:
                sm[k] = 1/Em[k]
                qm[k] = (k+1)**2/Em[k]
                Am[k] = (((k+1)**2)/Em[k]) * np.outer(ym[k], ym[k])
                bm[k] = (((k+1)**2)/Em[k])* ym[k]
            else:
                sm[k] = sm[k-1] + 1/Em[k]
                qm[k] = qm[k-1] + (k+1)**2/Em[k]
                Am[k] = Am[k-1] + (((k+1)**2)/Em[k]) * np.outer(ym[k], ym[k])
                bm[k] = bm[k-1] + (((k+1)**2)/Em[k])*ym[k]
            if k in index:
                const = (k+1)**2*sm[k]
                const = 1/const
                temp  = const * (Am[k] - np.outer(ym[k], bm[k])
                                            - np.outer(bm[k], ym[k]) + qm[k]*np.outer(ym[k], ym[k]))
                Vm.append(np.sqrt(np.diag(temp)))
                
    Vm = np.array(Vm)
    return Vm

class Timer:
    """Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`sec_minibatch_sgd`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()