import numpy as np
from functions import *
import sys
import multiprocessing as mp
from multiprocessing import Pool, get_context # Work on MacBook Air (M1, 2020, 8 cores).
import time 
import datetime
import logging
from gllim import GLLiM    

# We are now ready to install packages using R’s own function install.package:
# import rpy2's package module
import rpy2
# print(rpy2.__version__)
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr

# import R's utility package
utils = rpackages.importr('utils')

# select a mirror for R packages
utils.chooseCRANmirror(ind=32) # select the mirror in the list

# R package names
# packnames = ('utils', 'base', 'meteorits')
packnames = ('utils', 'base')

# R vector of strings
from rpy2.robjects.vectors import StrVector

# Selectively install what needs to be install.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

# import R's "utils" package    
utils = importr('utils')

# import R's "base" package
base = importr('base')
# Working with Object-Oriented Programming can be achieved in R: RS4
dollar = base.__dict__["$"]    

# import R's "meteorits" package:
    # meteorits: Mixture-of-Experts Modeling for Complex Non-Normal Distributions
    # https://cran.r-project.org/web/packages/meteorits/index.html

# utils.chooseCRANmirror(ind=32) # select the first mirror in the list Lyon.
# utils.install_packages('meteorits')
# meteorits = importr('meteorits') 

## Install meteoritsSim package for simulation.
# packnames = ('utils', 'base', 'meteorits')

packnamesDev = ('meteoritsSim')

# Selectively install what needs to be install.
names_to_install = [x for x in packnamesDev if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    devtools = utils.install_packages('devtools')

    d = {'package.dependencies': 'package_dot_dependencies',
         'package_dependencies': 'package_uscore_dependencies'}
    custom_analytics = importr('devtools', 
                       robject_translations = d)

    custom_analytics.install_github("Trung-TinNGUYEN/meteoritsSim", force = True)
    # custom_analytics.install_github("Trung-TinNGUYEN/meteoritsSim")
    
meteoritsSim = importr('meteoritsSim') 

import rpy2.robjects as robjects
r = robjects.r

start_time = time.time() # Calculate the runtime of a programme in Python.

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default=1, type=int, help='Type number.')
parser.add_argument('-K', '--K', default=2, type=int, help='Number of mixture components.')
parser.add_argument('-r' ,'--reps', default=1, type=int, help='Number of replications per sample size.')
parser.add_argument('-np','--nproc', default=8, type=int, help='Number of processes to run in parallel.') # Work on MacBook Air (M1, 2020, 8 cores)
parser.add_argument('-mi','--maxit', default=2000, type=int, help='Maximum EM iterations.')
parser.add_argument('-e', '--eps', default=1e-6, type=float, help='EM stopping criterion.')
parser.add_argument('-nnum', '--n_num', default=200, type=int, help='Number of different choices of sample size.')
parser.add_argument('-nsmax', '--ns_max', default=1000, type=int, help='Number of sample size maximum.')
parser.add_argument('-nsmin', '--ns_min', default=100, type=int, help='Number of sample size maximum.')
parser.add_argument('-ntries', '--n_tries', default=1, type=int, help='Number of trials for emNMoE of meteoritsSim package.')
parser.add_argument('-errorbar', '--errorbar', default=0, type=int, help='Number of trials for emNMoE of meteoritsSim package.')
parser.add_argument('-verbose', '--verbose', default=0,\
                    type=int, help='Log-likelihood should be printed or not during EM iterations.')
parser.add_argument('-verboseIRLS', '--verbose_IRLS', default=0,\
                    type=int, help='Criterion optimized by IRLS should be printed or not at each step of the EM algorithm.')

args = parser.parse_args()

print(args)

model        = args.model                    # Type number
K            = args.K                            # Number of mixture components
n_proc       = args.nproc                   # Number of cores to use
reps         = args.reps                      # Number of replications to run per sample size
max_iter     = args.maxit                 # Maximum EM iterations
eps          = args.eps                        # EM Stopping criterion.
n_num        = args.n_num                    # Number of different choices of sample size.
ns_max       = args.ns_max                    # Number of sample size maximum.
ns_min       = args.ns_min                    # Number of sample size minimum.
n_tries      = args.n_tries                    # Number of trials for emNMoE of meteoritsSim package
errorbar     = args.errorbar  
verbose      = args.verbose                    
verbose_IRLS = args.verbose_IRLS


exec(open("models.py").read())

logging.basicConfig(filename='std_mod' + str(model) + '_K' + str(K) +\
                    '.log', filemode='w', format='%(asctime)s %(message)s')
    
    
if (Ks == K):
    ns = np.concatenate([np.linspace(ns_min, 9*ns_min, 5), np.linspace(10*ns_min, ns_max, n_num-5)])
else:
    ns = np.linspace(ns_min, ns_max, n_num)

## Test.
## Test for 100 different choices of n between 10^2 and 10^5 on MacBook Air (M1, 2020, 8 cores). 
# ns = np.concatenate([np.linspace(100, 1000, 20), np.linspace(1000, ns_max, n_num-20)])
# ns = np.concatenate([np.linspace(ns_min, 9*ns_min, 5), np.linspace(10*ns_min, ns_max, n_num-5)])
# ns = np.concatenate([np.linspace(ns_min, 9*ns_min, 10), np.linspace(10*ns_min, ns_max, n_num-10)])
# ns = np.linspace(ns_min, ns_max, n_num)
# ns = np.concatenate([np.linspace(100, 900, 5), np.linspace(1000, 100000, n_num-5)]) 
# ns = np.concatenate([np.linspace(100, 900, 5), np.linspace(1000, 100000, n_num-5)]) 
# print(ns)
# print("Chose Model " + str(model))
# print(model)


# Main EM algorithm.   

def process_chunk_SGaME(bound):
    """ Run EM on a range of sample sizes. """
    ind_low = bound[0]
    ind_high= bound[1]

    m = ind_high - ind_low

    seed_ctr = 2023 * ind_low   # Random seed
    
    chunk_beta     = np.empty((m, reps, d+1,K-1))
    chunk_A        = np.empty((m, reps, d+1, K))
    chunk_Sigma    = np.empty((m, reps, 1, K))    

    for i in range(ind_low, ind_high):
        n = int(ns[i])

        for rep in range(reps):
            
            ## Test.
            # seed_ctr = 2023
            # n = 100
            
            # np.random.seed(seed_ctr)
            # set_seed = robjects.r('set.seed')
            # set_seed(seed_ctr)
            
            (betas, As, Sigmas, betasr, Asr, Sigmasr) = get_params(n)
            
            # Sample from the Softmax MoE models. 
            X = robjects.FloatVector(np.linspace(0,1, num = n))
            Y = meteoritsSim.sampleUnivNMoE(alphak = betasr, betak = Asr,\
                                                sigmak = Sigmasr, x = X)[0]
            
            nmoe = meteoritsSim.emNMoE(X, Y, K, d, l, n_tries, max_iter, eps,\
                                       verbose, verbose_IRLS, update_IRLS = True, nearTrue = True, \
                                           alphak_0 = betasr, betak_0 = Asr, sigmak_0 = Sigmasr)

            param = dollar(nmoe,"param")
            alpha = dollar(param,"alpha")
            beta = dollar(param,"beta")
            Sigma = dollar(param,"sigma2")
            
            # print('beta = \n', alpha)
            # print('A = \n', beta)
            # print('Sigma = \n', Sigma)
            
            #rpy2: Convert FloatVector or Matrix back to a Python array.
            alpha = np.asarray(alpha)
            beta  = np.asarray(beta)
            Sigma  = np.asarray(Sigma)
            
            # print('alpha.shape = ', alpha.shape)
            # print('beta.shape = ', beta.shape)
            # print('Sigma.shape = ', Sigma.shape)    
            
                        
            logging.warning('Model ' + str(model) + ', rep:' + str(rep) +\
                            ', n:' + str(n) + ", nind:" + str(i))
            
            chunk_beta[i-ind_low, rep, :, :]        = alpha
            chunk_A[i-ind_low, rep, :, :]           = beta
            chunk_Sigma[i-ind_low, rep, :, :]       = Sigma
             
                
            # # Sample from the mixture. 
            # X, Y = sample_SGaMEs(n, seed_ctr)

            # np.random.seed(seed_ctr+1)
            # (beta_start, A_start, Sigma_start) = init_params_SGaME(n,K)
            
            # # Using em_SGaME via a partition of starting values near the true components.
            # out = em_SGaME(X, Y, beta_start, A_start, Sigma_start, max_iter=max_iter, eps=eps)
            
            # logging.warning('Model ' + str(model) + ', rep:' + str(rep) +\
            #                 ', n:' + str(n) + ", nind:" + str(i) + ", iters:" + str(out[-2]))
                        
            # chunk_beta[i-ind_low, rep, :, :]        = out[0]
            # chunk_A[i-ind_low, rep, :, :]           = out[1]   
            # chunk_Sigma[i-ind_low, rep, :, :]       = out[2]    
            # chunk_iters[i-ind_low, rep]             = out[3]   

            seed_ctr += 1

    return (chunk_beta, chunk_A, chunk_Sigma)

# Multiprocessing.

proc_chunks_SGaME = []

## Uniform distribution.
Del = n_num // n_proc 

for i in range(n_proc):
    if i == n_proc-1:
        proc_chunks_SGaME.append(( (n_proc-1) * Del, n_num) )

    else:
        proc_chunks_SGaME.append(( (i*Del, (i+1)*Del ) ))

if n_proc == 1: # For quick test.
    proc_chunks_SGaME = [(99, 100)]
    
elif ((n_proc == 8) & (n_num == 100)): # 8 Cores n_num = 100
    proc_chunks_SGaME = [(0, 25), (25, 40), (40, 55), (55, 70), (70, 82), (82, 90),\
                    (90, 96), (96, 100)] # 8 Cores for 100 different choices of n.
        
elif ((n_proc == 8) & (n_num == 150)): # 8 Cores n_num = 100
    proc_chunks_SGaME = [(0, 50), (50, 90), (90, 110), (110, 125), (125, 137), (137, 143),\
                    (143, 148), (148, 150)] # 8 Cores for 100 different choices of n.        
        
elif ((n_proc == 8) & (n_num == 200)): # 8 Cores n_num = 200
    proc_chunks_SGaME = [(0, 80), (80, 120), (120, 150), (150, 170), (170, 188), (188, 194),\
                    (194, 198), (198, 200)] # 8 Cores for 100 different choices of n.

elif n_proc == 12: # 12 Cores
    proc_chunks_SGaME = [(0, 25), (25, 40), (40, 50), (50, 60), (60, 67), (67, 75),\
                    (75, 80), (80, 85), (85, 90), (90, 94), (94, 97), (97, 100)]        
else:
    print("Please modify proc_chunks according to the core of your computer.!")
    
with get_context("fork").Pool(processes=n_proc) as pool:  # Work on MacBook Air (M1, 2020).
    proc_results_SGaME = [pool.apply_async(process_chunk_SGaME,
                                      args=(chunk,))
                    for chunk in proc_chunks_SGaME]

    result_chunks_SGaME = [r.get() for r in proc_results_SGaME]

# Save the result for SGaME models.
done_beta    = np.concatenate([result_chunks_SGaME[j][0] for j in range(n_proc)], axis=0)
done_A       = np.concatenate([result_chunks_SGaME[j][1] for j in range(n_proc)], axis=0)
done_Sigma   = np.concatenate([result_chunks_SGaME[j][2] for j in range(n_proc)], axis=0)

np.save("results_SGaME/result_model" + str(model) +"_K" + str(K) +"_ns_min" +\
        str(int(ns[0])) +"_ns_max" + str(int(ns[-1])) + "_n_num" + str(n_num) + \
            "_rep"  + str(reps)+ "_ntries" + str(n_tries) + "_beta.npy", done_beta)

np.save("results_SGaME/result_model" + str(model) +"_K" + str(K) +"_ns_min" +\
        str(int(ns[0])) +"_ns_max" + str(int(ns[-1])) + "_n_num" + str(n_num) + \
            "_rep"  + str(reps)+ "_ntries" + str(n_tries) + "_A.npy", done_A)    

np.save("results_SGaME/result_model" + str(model) +"_K" + str(K) +"_ns_min" +\
        str(int(ns[0])) +"_ns_max" + str(int(ns[-1])) + "_n_num" + str(n_num) + \
            "_rep"  + str(reps)+ "_ntries" + str(n_tries) + "_Sigma.npy", done_Sigma)      
    

print("--- %s seconds ---" % (time.time() - start_time))


