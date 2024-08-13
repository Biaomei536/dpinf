import numpy as np
from matplotlib import pyplot as plt
import os
from scipy.stats import laplace
from tool_gradient import CustomGradient
from tool_function import*
import scipy.stats as stats
random_state = 68
import os
import h5py
np.random.seed(random_state)

'''

This script is used for estimating population quantiles under ldp guarantee.

'''

def generate_schedule(T=5000, Em= 5):
    
    warm_out = int(T*0.05)
    iter_schedule = np.ones(T)
    iter_schedule[warm_out: ] = np.ones(T-warm_out, dtype='int') * Em
    iter_schedule = iter_schedule.astype(int)
    # if Em == 'C5':
    #     iter_schedule[warm_out: ] = np.ones(T-warm_out, dtype='int') * 5
    # elif Em =='C1':
    #     iter_schedule[warm_out: ] = np.ones(T-warm_out, dtype='int')
    # elif Em =='C3':
    #     iter_schedule[warm_out: ] = np.ones(T-warm_out, dtype='int') * 3
    # elif Em == 'P(1/3)':
    #     iter_schedule[warm_out: ] = np.ceil(np.power(temp, 1/3)).astype(int)
    # elif Em == 'P(1/2)':
    #     iter_schedule[warm_out: ] = np.ceil(np.power(temp, 1/2)).astype(int)
    # elif Em == 'P(2/3)':
    #     iter_schedule[warm_out: ] = np.ceil(np.power(temp, 2/3)).astype(int)
    # else:
    #     print('Sequences for Em have not been realized')
    return iter_schedule

class Client:
    def __init__(self, true_theta, mode = 'logistic', tau = np.array([0.10, 0.25, 0.50, 0.75, 0.90])):
        
        self.true_theta = true_theta
        self.dim = len(true_theta)
        self.theta = None # represent local - iterate
        self.mode = mode
        self.grad = CustomGradient(mode = mode)
        self.tau = tau # used for quantile regression and quantile estimator
        self.tau_value = stats.norm.ppf(self.tau, loc=0, scale=1)
        self.P = None

    def generate_iter(self, n1, n2, Em):
        
        total_num = n1 + n2 * Em
        X = np.random.normal(loc=0, scale=1, size = (total_num, self.dim))
        
        if self.P != None:
            noise = np.random.binomial(1, self.P, (total_num, self.dim))

        if self.P == None:
            self.data_iter_warm_out = data_iter(1, X[:n1])
            self.train_iter = data_iter(Em, X[n1:])
        else:
            self.data_iter_warm_out =data_iter(1, X[:n1], noise[:n1])
            self.train_iter = data_iter(Em, X[n1:], noise[n1:])
            
    def ldp_sgd(self, lr, num):
        if self.P == None:

            if num == 1:
                Xs = next(self.data_iter_warm_out)
            else:
                Xs = next(self.train_iter)[0]

            for k, x in enumerate(Xs):
                rr = np.where(x<self.theta, 1, 0)
                gd = -self.tau + rr
                self.theta = self.theta - lr * gd/num
        else:

            if num == 1:
                Xs,  Ns  = next(self.data_iter_warm_out)
            else:
                Xs,  Ns  = next(self.train_iter)
            for k, (x, noise) in enumerate(zip(Xs, Ns)):
                rr = np.where(x<self.theta, noise, 1-noise)
                gd = -self.tau +  (rr-(1-self.P))/(2*self.P-1)
                self.theta = self.theta - lr * gd/num
            
class Aggregator:
    def __init__(self, communication_rounds, theta0, alpha, mode, omega, epsi, clients, Em, lr0):

        '''
        mode = logistic, linear, quantile, quantile_reg
        '''
        self.theta0 = theta0
        self.param_dim = len(theta0)
        self.num_rounds = 0
        self.clients = clients
        self.num_clients = len(clients)
        self.communication_rounds = communication_rounds
        self.Em = Em
        self.iter_schedule = generate_schedule(T=communication_rounds, Em = Em)
        self.lr0 = lr0
        self.theta_bar = []
        self.alpha = alpha
        self.omega = omega
        self.warm_out = round(0.05 * self.communication_rounds)
        self.Vt = None
        self.mode = mode
        self.epsi = epsi
        self.P = None
        if self.epsi != None:
            self.set_lap_param()

    def initialization(self):
        n = round(self.communication_rounds - self.warm_out)
        for k, client in enumerate(self.clients):
            client.theta = self.theta0
            client.P = self.P
            client.generate_iter(self.warm_out, n, self.Em)
    
    def set_lap_param(self):
        self.P = sigmoid(self.epsi)

    def iter(self):
        self.lr = self.lr0 / np.power(self.num_rounds+1, self.alpha)
        for client in self.clients:
            client.ldp_sgd(self.lr, self.iter_schedule[self.num_rounds])
        
    
    def synchronization(self):
        client_theta = np.zeros((self.num_clients, self.param_dim))
        for k, client in enumerate(self.clients):
            client_theta[k] = client.theta.flatten()
            
        self.theta_bar.append(np.average(client_theta, axis=0, weights=self.omega))
        self.broadcast()
        if self.num_rounds >= self.warm_out:
            self.update_Vt()
        self.num_rounds = self.num_rounds+1

    def broadcast(self):
        for client in self.clients:
            client.theta = self.theta_bar[-1]

    def update_Vt(self):
        current_schedule = self.iter_schedule[self.num_rounds]
        current_rounds = self.num_rounds - self.warm_out + 1
        if current_rounds == 1:
            self.sm = [1/current_schedule]
            self.qm = [current_rounds**2/ current_schedule]
            self.ym = [self.theta_bar[-1]]
        else:
            self.sm.append(self.sm[-1] + 1/current_schedule)
            self.qm.append(self.qm[-1] + ((current_rounds**2)/ current_schedule))
            self.ym.append(((current_rounds-1)/current_rounds) * self.ym[-1] 
                           + (self.theta_bar[-1]/current_rounds) )

    def figure_Ct(self):
        current_rounds = self.num_rounds - self.warm_out
        Constant = (current_rounds **2) * self.sm[-1]
        Constant = 1/ Constant

        iter_schedule = self.iter_schedule[self.warm_out:]
        
        A = 0
        b = 0
        for k in range(current_rounds):
            temp1 = (((k+1)**2)/iter_schedule[k])
            A += temp1 * np.outer(self.ym[k], self.ym[k])
            b += temp1 * self.ym[k] 
        temp = Constant * (A - np.outer(self.ym[-1], b) - np.outer(b, self.ym[-1]) + self.qm[-1] * np.outer(self.ym[-1], self.ym[-1]))
        self.Vt = np.sqrt(np.diag(temp))

    def construct_CI(self, q=6.811):
        if self.Vt == None:
            self.figure_Ct()
        Std = self.Vt
        low = self.ym[-1] - q * Std
        upper = self.ym[-1] + q * Std
        self.CI = np.vstack((low, upper))
        return self.CI
    
def main(tau, alpha, dim, mode, epsi, communication_rounds, lr0, Em, Num_of_clients):

    global_theta = stats.norm.ppf(tau, loc=0, scale=1)
    theta0 = np.random.normal(loc=0, scale=1, size=dim) 
    omega = np.ones(Num_of_clients)/ Num_of_clients

    clients = []
    for k in range(Num_of_clients):
        clients.append(Client(global_theta, mode, tau))
    
    aggregator = Aggregator(communication_rounds, theta0, alpha, mode, omega, epsi, clients, Em, lr0)
    aggregator.initialization()

    for m in range(communication_rounds):
        aggregator.iter()
        aggregator.synchronization()

    return np.array(aggregator.ym), global_theta

                                           
                                                        
alpha = 0.505                                                         # decreasing rate of learning rate                                                  
Em = 5                                                                # E_m                               
tau = np.array([0.10, 0.25, 0.50, 0.75, 0.90])                        # number of clients  
Num_of_clients = 8                                                    # weights of clients
dim = len(tau)                                                        # dimension of parameter
omega = np.ones(Num_of_clients) / Num_of_clients                      # initial learning rate: gamma_0
lr0 = 0.4 
communication_rounds = round(2e5) 
mode = 'quantile'
repetition = 5
epsi = 1 # None, 0.5, 1, 3.

iter_schedule = generate_schedule(T=communication_rounds, Em = Em) 
theta_res = np.zeros((repetition, dim))
Path = 'result'
current_path = os.path.join(Path, mode)
os.makedirs(current_path, exist_ok=True)


true_params = np.zeros((repetition, dim))
toal_rounds = round(communication_rounds * 0.95)
print('The total rounds is:', toal_rounds)

num_index = 200
indexs = np.geomspace(100, toal_rounds, num_index).astype(int)-1
res_yms = np.zeros((repetition, num_index, dim))
res_Vms = np.zeros((repetition, num_index, dim))
timer = Timer()
timer.start()
for rep in range(repetition):
    res_ym, true_param = main(tau, alpha, dim, mode, epsi, 
                                 communication_rounds, lr0, Em, Num_of_clients)
    true_params[rep] = true_param
    Vm = get_index_CI(res_ym, indexs)
    res_yms[rep] = res_ym[indexs][:]
    res_Vms[rep] = Vm

    if rep% 20 == 0:
            print(f'The {rep}-th experiment has been accomplished:')

res = np.zeros((num_index, res_yms.shape[2]))
for k in range(num_index):
    CI = [res_yms[:,k,:]-6.811*res_Vms[:,k,:], res_yms[:,k,:]+6.811*res_Vms[:,k,:]]
    a = (true_params<CI[1]).astype(int)
    b = (true_params>CI[0]).astype(int)
    c = a*b
    res[k] = np.mean(c,axis=0)
print('Coverage rate:', res[-1])

with h5py.File(os.path.join(current_path, f'res{epsi}_lr0_{lr0}.h5'), 'w') as f:
    f.create_dataset('ym', data=res_yms)
    f.create_dataset('Vm', data=res_Vms)
    f.create_dataset('true_params', data=true_params)

timer.stop()
print('Time:', timer.sum())
