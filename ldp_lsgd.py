import numpy as np
from matplotlib import pyplot as plt
import os
from scipy.stats import laplace
from tool_gradient import CustomGradient
from tool_function import*
import scipy.stats as stats
from scipy.stats import truncnorm
import os
import h5py
random_state = 68
np.random.seed(random_state)

'''

This script is used for conducting statistical inference under ldp guarantee.

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
    def __init__(self, true_theta, mode = 'logistic', tau = 0.5):
        
        self.true_theta = true_theta
        self.dim = len(true_theta)
        self.theta = None # represent local - iterate
        self.mode = mode
        self.grad = CustomGradient(mode = mode)
        self.tau = tau # used for quantile regression and quantile estimator
        self.tau_value = stats.norm.ppf(self.tau, loc=0, scale=1)
        self.lap_param = None
        self.mu = None

    def generate_iter(self, n1, n2, Em):
        total_num = n1 + n2 * Em
        X = np.random.uniform(low = self.mu - 0.5, high = self.mu + 0.5, size = (total_num, self.dim))
        if self.lap_param != None:
            noise = laplace.rvs(0, self.lap_param, (total_num, self.dim)) 
        if self.mode == 'logistic':
            p = sigmoid(X @ self.true_theta)
            y = np.random.binomial(1, p, total_num)
        elif self.mode == 'linear_reg':
            y = X @ self.true_theta
            y += np.random.normal(0, 1, total_num)
        elif self.mode == 'quantile_reg':
            # 分位数回归/中位数回归
            y = X @ self.true_theta
            y += np.random.normal(0, 1, total_num) - self.tau_value
        if self.lap_param == None:
            self.data_iter_warm_out = data_iter(1, X[:n1], y[:n1])
            self.train_iter = data_iter(Em, X[n1:], y[n1:])
        else:
            self.data_iter_warm_out = data_iter(1, X[:n1], y[:n1], noise[:n1])
            self.train_iter = data_iter(Em, X[n1:], y[n1:], noise[n1:])

    def ldp_sgd(self, lr, num):
        if self.lap_param == None:
            if num ==1:
                Xs, Ys = next(self.data_iter_warm_out)
            else:
                Xs, Ys = next(self.train_iter)
            for k, (x, y) in enumerate(zip(Xs, Ys)):
                gd = self.grad(self.theta, x, y) 
                self.theta = self.theta - lr * gd/num
        else:   
            if num == 1:
                Xs, Ys, Ns = next(self.data_iter_warm_out)
            else:
                Xs, Ys, Ns = next(self.train_iter)

            for k, (x, y, noise) in enumerate(zip(Xs, Ys, Ns)):
                gd = self.grad(self.theta, x, y) 
                if self.lap_param != None:
                    gd += noise
                self.theta = self.theta - lr * gd/num
            
class Aggregator:
    def __init__(self, communication_rounds, theta0, alpha, mode, omega, epsi, clients, Em, lr0):

        '''
        mode = logistic, linear_reg,  quantile_reg
        '''
        self.theta0 = theta0
        self.param_dim = len(theta0)
        self.num_rounds = 0
        self.clients = clients
        self.num_clients = len(clients)
        self.communication_rounds = communication_rounds
        self.iter_schedule = generate_schedule(T=communication_rounds, Em = Em)  
        self.Em = Em
        self.lr0 = lr0
        self.theta_bar = []
        self.alpha = alpha
        self.omega = omega
        self.warm_out = round(0.05 * communication_rounds)
        self.Vt = None # \wilde{\Pi}_t
        self.mode = mode
        self.epsi = epsi
        self.Em = Em
        self.lap_param = None
        self.mu = truncnorm.rvs(-0.5, 0.5, loc=0, scale=1, size=self.num_clients)
        # self.mu = np.zeros(self.num_clients)
        if self.epsi != None:
            self.set_lap_param()

    def initialization(self):
        
        n = round(self.communication_rounds - self.warm_out)
        for k, client in enumerate(self.clients):
            client.theta = self.theta0
            if self.epsi != None:
                client.lap_param = self.lap_param[k]
            
            client.mu = self.mu[k]
            client.generate_iter(self.warm_out, n, self.Em)

    def set_lap_param(self):
        if self.mode == 'logistic':
            self.lap_param = 2 * (0.5 + np.abs(self.mu)) * (self.param_dim / self.epsi)
        
        elif self.mode == 'linear_reg':
            self.lap_param = 2 * (0.5 + np.abs(self.mu)) * (self.param_dim / self.epsi)
            # sigma = 1
        elif self.mode == 'quantile_reg':
            self.lap_param = (0.5 + np.abs(self.mu)) * self.param_dim / self.epsi

    def iter(self):
        self.lr = self.lr0 / np.power(self.num_rounds+1, self.alpha)
        for client in self.clients:
            client.ldp_sgd(self.lr, self.iter_schedule[self.num_rounds])
        
    
    def synchronization(self):
        client_theta = np.zeros((self.num_clients, self.param_dim))
        for k, client in enumerate(self.clients):
            client_theta[k] = client.theta
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
    
def main(alpha, dim, mode, epsi, communication_rounds, lr0, Em, Num_of_clients):
    

    theta0 = np.random.normal(loc=0, scale=1, size=dim)
    # initial theta 
    temp = np.random.normal(loc=0, scale=1, size=(1,dim))
    local_thetas = np.outer(np.ones(Num_of_clients), temp)
    omega = np.ones(Num_of_clients)/ Num_of_clients
    global_theta = np.average(local_thetas, axis=0, weights=omega)
    # true global theta

    clients = []
    for k in range(Num_of_clients):
        clients.append(Client(local_thetas[k], mode))
    aggregator = Aggregator(communication_rounds, theta0, alpha, mode, omega, epsi, clients, Em, lr0)
    aggregator.initialization()
    for m in range(communication_rounds):
        aggregator.iter()
        aggregator.synchronization()

    return np.array(aggregator.ym), global_theta

                                           
#parameters                                           
alpha = 0.505                                                                   # decreasing rate of learning rate                                                                                                            
Em = 5                                                                          # E_m
Num_of_clients = 8                                                              # number of clients                                   
omega = np.ones(Num_of_clients)/ Num_of_clients                                 # weights of clients
dim = 4                                                                         # dimension of parameter
lr0 = 2.0                                                                       # initial learning rate: gamma_0                                               
communication_rounds = round(3e4)                                               
# total communication rounds           
mode = 'linear_reg'                                                             # logistic/quantile_reg/linear_reg                   
repetition = 1                                                                  # number of repetition                  
epsi = 3                                                                     # privacy budget                                    
# epsi = None means no privacy guarantee
Path = 'result'      
current_path = os.path.join(Path, mode)                                                           
os.makedirs(current_path, exist_ok=True)


# evaluate coverage rates and length of confidence intervals of the proposed inference procedure.
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
    res_ym, true_param = main(alpha, dim, mode, epsi, 
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
