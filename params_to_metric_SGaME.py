import sys
import numpy as np
from discrepancies import *
from scipy.spatial.distance import cdist
import logging

# We are now ready to install packages using R’s own function install.package:
# import rpy2's package module
import rpy2
# print(rpy2.__version__)

import rpy2.robjects as robjects
r = robjects.r


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

    
dists = np.empty((n_num, reps))



### Old version: 2023-05-18    
# beta_list_n0    = np.load("results_SGaME/result_model" + str(model) +"_K" + str(K) +"_n" +\
#         str(int(ns[-1])) +"_rep"  + str(reps)+ "_ntries" + str(n_tries) + "_beta.npy")

# # Concatenate the last column with zero values.    
# beta_list_nlast = np.zeros((beta_list_n0.shape[0],beta_list_n0.shape[1],beta_list_n0.shape[2],1))
# beta_list = np.concatenate((beta_list_n0, beta_list_nlast), axis = -1) 
   
# A_list     = np.load("results_SGaME/result_model" + str(model) +"_K" + str(K) +"_n" +\
#         str(int(ns[-1])) +"_rep" + str(reps)+ "_ntries" + str(n_tries)+ "_A.npy")
# Sigma_list = np.load("results_SGaME/result_model" + str(model) +"_K" + str(K) +"_n" +\
#         str(int(ns[-1])) +"_rep" + str(reps) + "_ntries" + str(n_tries)+ "_Sigma.npy")

beta_list_n0    = np.load("results_SGaME/result_model" + str(model) +"_K" + str(K) +"_ns_min" +\
                          str(int(ns[0])) +"_ns_max" + str(int(ns[-1])) + "_n_num" + str(n_num) + \
                              "_rep"  + str(reps)+ "_ntries" + str(n_tries) + "_beta.npy")

    
# Concatenate the last column with zero values.    
beta_list_nlast = np.zeros((beta_list_n0.shape[0],beta_list_n0.shape[1],beta_list_n0.shape[2],1))
beta_list = np.concatenate((beta_list_n0, beta_list_nlast), axis = -1) 

A_list    = np.load("results_SGaME/result_model" + str(model) +"_K" + str(K) +"_ns_min" +\
                          str(int(ns[0])) +"_ns_max" + str(int(ns[-1])) + "_n_num" + str(n_num) + \
                              "_rep"  + str(reps)+ "_ntries" + str(n_tries) + "_A.npy")

Sigma_list    = np.load("results_SGaME/result_model" + str(model) +"_K" + str(K) +"_ns_min" +\
                          str(int(ns[0])) +"_ns_max" + str(int(ns[-1])) + "_n_num" + str(n_num) + \
                              "_rep"  + str(reps)+ "_ntries" + str(n_tries) + "_Sigma.npy")


    
    
for i in range(n_num):
    (betas, As, Sigmas, betasr, Asr, Sigmasr) = get_params(ns[i])
    
    for j in range(reps):
        if model == 1:
            dists[i,j] = loss_SGaME_exact(betas, As, Sigmas**2, beta_list[i,j,:,:],\
                                          A_list[i,j,:,:], Sigma_list[i,j,:,:])
             
        else:
            sys.exit("Model unrecognized.")  



### Old version: 2023-05-18    
# np.save("results_SGaME/result_model" + str(model) +"_K" + str(K) +"_n" +\
#         str(int(ns[-1])) +"_rep" + str(reps) + "_ntries" + str(n_tries)+ "_loss.npy", dists)

np.save("results_SGaME/result_model" + str(model) +"_K" + str(K) +"_ns_min" +\
        str(int(ns[0])) +"_ns_max" + str(int(ns[-1])) + "_n_num" + str(n_num) + \
            "_rep"  + str(reps)+ "_ntries" + str(n_tries) +  "_loss.npy", dists)    
    
    
# np.save("results_SGaME/result_model" + str(model) +"_K" + str(K) +"_n" +\
#         str(int(ns[-1])) +"_rep" + str(reps)+ "_loss.npy", dists)
    
# dists = np.empty((n_num, reps))

# pis    = np.load("results_SGaME/result_model" + str(model) +"_K" + str(K) +\
#                  "_n" + str(int(ns[-1])) +"_rep" + str(reps)+ "_pi.npy")
# cs     = np.load("results_SGaME/result_model" + str(model) +"_K" + str(K) +\
#                  "_n" + str(int(ns[-1])) +"_rep" + str(reps)+ "_c.npy")
# Gammas = np.load("results_SGaME/result_model" + str(model) +"_K" + str(K) +\
#                  "_n" + str(int(ns[-1])) +"_rep" + str(reps)+ "_Gamma.npy")
# As     = np.load("results_SGaME/result_model" + str(model) +"_K" + str(K) +\
#                  "_n" + str(int(ns[-1])) +"_rep" + str(reps)+ "_A.npy")
# bs     = np.load("results_SGaME/result_model" + str(model) +"_K" + str(K) +\
#                  "_n" + str(int(ns[-1])) +"_rep" + str(reps)+ "_b.npy")
# nus    = np.load("results_SGaME/result_model" + str(model) +"_K" + str(K) +\
#                  "_n" + str(int(ns[-1])) +"_rep" + str(reps)+ "_nu.npy")

# for i in range(n_num):
#     theta0, sigma0, pi0, c0, Gamma0, A0, b0, nu0 = get_params(ns[i])
    
#     for j in range(reps):
#         if fitGLLiM == 0:
#             if model == 1:
#                 dists[i,j] = gauss_loss_SGaME1(pi0, c0, Gamma0, A0, b0, nu0, pis[i,j,:],\
#                                         cs[i,j,:,:], Gammas[i,j,:,:,:], As[i,j,:,:,:],\
#                                             bs[i,j,:,:], nus[i,j,:,:,:])
#             elif model == 2:
#                 dists[i,j] = gauss_loss_SGaME2(pi0, c0, Gamma0, A0, b0, nu0, pis[i,j,:],\
#                                         cs[i,j,:,:], Gammas[i,j,:,:,:], As[i,j,:,:,:],\
#                                             bs[i,j,:,:], nus[i,j,:,:,:])         
#             else:
#                 sys.exit("Model unrecognized.")        
#         else:    
#             dists[i,j] = gauss_loss_SGaME1(pi0, c0, Gamma0, A0, b0, nu0, pis[i,j,:],\
#                                     cs[i,j,:,:], Gammas[i,j,:,:,:], As[i,j,:,:,:],\
#                                         bs[i,j,:,:], nus[i,j,:].reshape(K, l, l))

# np.save("results_SGaME/result_model" + str(model) +"_K" + str(K) +"_n" +\
#         str(int(ns[-1])) +"_rep" + str(reps)+ "_loss.npy", dists)    
