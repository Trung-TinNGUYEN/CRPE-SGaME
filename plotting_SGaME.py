import numpy as np
from functions import *
import matplotlib.pyplot as plt
import matplotlib
import logging

print("Log-log scale plots for the simulation results of CRPE-SGaME")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default=1, type=int, help='Type number.')
parser.add_argument('-K', '--K', default=2, type=int, help='Number of mixture components.')
parser.add_argument('-r' ,'--reps', default=40, type=int, help='Number of replications per sample size.')
parser.add_argument('-np','--nproc', default=8, type=int, help='Number of processes to run in parallel.') # Work on MacBook Air (M1, 2020, 8 cores)
parser.add_argument('-mi','--maxit', default=2000, type=int, help='Maximum EM iterations.')
parser.add_argument('-e', '--eps', default=1e-6, type=float, help='EM stopping criterion.')
parser.add_argument('-nnum', '--n_num', default=200, type=int, help='Number of different choices of sample size.')
parser.add_argument('-nsmax', '--ns_max', default=100000, type=int, help='Number of sample size maximum.')
parser.add_argument('-nsmin', '--ns_min', default=100, type=int, help='Number of sample size maximum.')
parser.add_argument('-ntries', '--n_tries', default=1, type=int, help='Number of trials for emNMoE of meteoritsSim package.')
parser.add_argument('-errorbar', '--errorbar', default=0, type=int, help='Number of trials for emNMoE of meteoritsSim package.')
parser.add_argument('-verbose', '--verbose', default=0,\
                    type=int, help='Log-likelihood should be printed or not during EM iterations.')
parser.add_argument('-verboseIRLS', '--verbose_IRLS', default=0,\
                    type=int, help='Criterion optimized by IRLS should be printed or not at each step of the EM algorithm.')


args = parser.parse_args()

print(args)

model       = args.model                    # Type number
K           = args.K                            # Number of mixture components
n_proc      = args.nproc                   # Number of cores to use
reps        = args.reps                      # Number of replications to run per sample size
max_iter    = args.maxit                 # Maximum EM iterations
eps         = args.eps                        # EM Stopping criterion.
n_num       = args.n_num                    # Number of different choices of sample size.
ns_max      = args.ns_max                    # Number of sample size maximum.
ns_min     = args.ns_min                    # Number of sample size minimum.
n_tries      = args.n_tries                    # Number of trials for emNMoE of meteoritsSim package
errorbar      = args.errorbar  
verbose      = args.verbose                    
verbose_IRLS      = args.verbose_IRLS


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


text_size = 17

lw = 2
elw = 0.7

matplotlib.rc('text', usetex=True)
matplotlib.rc('xtick', labelsize=text_size) 
matplotlib.rc('ytick', labelsize=text_size) 

def plot_model(K, model, n0=0):
    
    ### Old version: 2023-05-18
    # D = np.load("results_SGaME/result_model" + str(model) +"_K" + str(K) +"_n" +\
    #         str(int(ns[-1])) +"_rep" + str(reps) + "_ntries" + str(n_tries)+ "_loss.npy")

    D = np.load("results_SGaME/result_model" + str(model) +"_K" + str(K) +"_ns_min" +\
            str(int(ns[0])) +"_ns_max" + str(int(ns[-1])) + "_n_num" + str(n_num) + \
                "_rep"  + str(reps)+ "_ntries" + str(n_tries) +  "_loss.npy")    
    
    # # ### Test code.
    # n0 = 0
    # D  = D[38:,:] 
    # # D = np.load("results_SGaME/result_model1_K3_n100000_rep20_ntries1_loss.npy")
    # # D = np.load("results_SGaME/result_model1_K3_n100000_rep40_ntries1_loss.npy")
    # # D = np.load("results_SGaME/result_model1_K3_ns_min100_ns_max100000_n_num100_rep20_ntries1_loss.npy")
    # # D = np.load("results_SGaME/result_model1_K2_ns_min100_ns_max100000_n_num100_rep20_ntries1_loss.npy")
    
    
    # # # Test sample sizes from 10^3-10^5 using n0=5
    # # loss        = np.mean(D[5:,:], axis=1)
    # # yerr        = 2*np.std(D[5:,:], axis=1)
    # # lab="temp"

    
    # ## Test: Remove outliers.
    # ## Test sample sizes from 10^2-10^5
    # fig = plt.figure()
    
    # loss        = np.mean(D, axis=1)
    # yerr        = 2*np.std(D, axis=1)
    
    # lab="temp"
    
    # normal = (yerr < 1)
    # loss = loss[normal]
    # yerr = yerr[normal]
    # Y = np.array(np.log(loss)).reshape([np.sum(normal),1])
    # if model == 1:
    #     label = "$\\overline{\mathcal{D}}(\widehat G_n, G_0)$"
    # else:
    #     label = "$\\widetilde{\mathcal{D}}(\widehat G_n, G_0)$"
    # # plt.errorbar(np.log(ns[normal]), Y[n0:].reshape([-1]), yerr=yerr[n0:], color='red', linestyle = '-', lw=lw, elinewidth=elw, label=label)    
    # plt.errorbar(np.log(ns[normal]), Y[n0:].reshape([-1]), color='red', linestyle = '-', lw=lw, elinewidth=elw, label=label)    
    
    # # TestT
    # X = np.empty([np.sum(normal), 2])
    # X.shape
    # X[:,0] = np.repeat(1, np.sum(normal))
    # X[:,1] = np.log(ns[normal])
    # Y = Y[n0:] 
    
    # beta = (np.linalg.inv(X.transpose() @ X) @ X.transpose() @ Y)
    # print("n0 = "+ str(n0) +", Beta = ", beta[1,0])

    # plt.plot(X[:,1], X @ beta, lw=lw, color='black', linestyle = '-.', label=str(np.round(beta[0,0], 1)) +\
    #           "$n^{" + str(np.round(beta[1,0],5)) + "}$" )
    
    # plt.xlabel("log(sample size)", fontsize=text_size)
    # plt.ylabel("log(loss)", fontsize=text_size)#"$\log$ " + lab)
    # plt.legend(loc="upper right", title="", prop={'size': text_size})
    
    
    #####
    
    # # Original.
    fig = plt.figure()
    
    loss        = np.mean(D, axis=1)
    yerr        = 2*np.std(D, axis=1)
    
    yerr_std  = 2*np.std(yerr)
    
    lab="temp"
    
    Y = np.array(np.log(loss)).reshape([-1,1])
    if ((model == 1) & (Ks == K)):
        label = "$\mathcal{D}_1(\widehat G_n, G_{*})$"
    else:
        label = "$\mathcal{D}_2(\widehat G_n, G_{*})$"
    
    if (errorbar == 1):
        plt.errorbar(np.log(ns[n0:]), Y[n0:].reshape([-1]), yerr=yerr[n0:], color='red', linestyle = '-', lw=lw, elinewidth=elw, label=label)
    else:
        plt.errorbar(np.log(ns[n0:]), Y[n0:].reshape([-1]), color='red', linestyle = '-', lw=lw, elinewidth=elw, label=label)

    # plt.errorbar(np.log(ns[n0:]), Y[n0:], label=label)
    # plt.grid(True, alpha=.5)
    
    # Original
    X = np.empty([ns.size-n0, 2])
    X[:,0] = np.repeat(1, ns.size-n0)
    X[:,1] = np.log(ns[n0:])
    Y = Y[n0:]     
    
    beta = (np.linalg.inv(X.transpose() @ X) @ X.transpose() @ Y)
    print("n0 = "+ str(n0) +", Beta = ", beta[1,0])

    plt.plot(X[:,1], X @ beta, lw=lw, color='black', linestyle = '-.', label=str(np.round(beta[0,0], 1)) +\
             "$n^{" + str(np.round(beta[1,0],5)) + "}$" )
    
    plt.xlabel("log(sample size)", fontsize=text_size)
    plt.ylabel("log(loss)", fontsize=text_size)#"$\log$ " + lab)
    plt.legend(loc="upper right", title="", prop={'size': text_size})


    plt.savefig("plots_SGaME/plot_model" + str(model) +"_K" + str(K) + "_n0_" +\
                str(n0) +"_ns_min" + str(int(ns[0])) +"_ns_max" +\
                    str(int(ns[-1])) + "_n_tries" + str(n_tries) + "_n_num" + str(n_num) +"_errorbar"+str(errorbar) +"_rep" + str(D.shape[1])+\
                    ".pdf", bbox_inches = 'tight',pad_inches = 0)
    
    ### Old version: 2023-05-18
    # plt.savefig("plots_SGaME/plot_model" + str(model) +"_K" + str(K) + "_n0_" +\
    #             str(n0) +"_n" + str(int(ns[-1])) +"_rep" + str(D.shape[1])+\
    #                 ".pdf", bbox_inches = 'tight',pad_inches = 0)

# print("Log-log scale plots for the simulation results of CRPE-GLLiM")
if (K > Ks):
    for i in range(0,50):
        plot_model(model= model, K= K, n0 = i)
else:
    plot_model(model= model, K= K, n0 = 0)