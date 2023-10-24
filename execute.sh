#!/bin/bash
         
#### Running on terminal:
# conda activate base --> cd ~CRPE-SGaME --> source execute.sh                   
#### Test for 100 different choices of n between 10^2 and 10^5 on MacBook Air (M1, 2020, 8 cores).


###############################################
### Upload overleaf 2023-05-21: Good results
###############################################
#Namespace(model=1, K=2, reps=40, nproc=8, maxit=2000, eps=1e-06, n_num=150, ns_max=100000, ns_min=100, n_tries=5, errorbar=False, verbose=True, verbose_IRLS=False)
#--- 877.0558261871338 seconds ---
# ns = np.concatenate([np.linspace(ns_min, 9*ns_min, 5), np.linspace(10*ns_min, ns_max, n_num-5)])
python experiment.py -m 1 -K 2 -r 40 -np 8 -nsmax 100000 -nsmin 100 -ntries 1 -nnum 200

python params_to_metric_SGaME.py -m 1 -K 2 -r 40 -np 8 -nsmax 100000 -nsmin 100 -ntries 1 -nnum 200

python plotting_SGaME.py -m 1 -K 2 -r 40 -np 8 -nsmax 100000 -nsmin 100 -ntries 1 -nnum 200 -errorbar True


###
#Namespace(model=1, K=3, reps=40, nproc=8, maxit=2000, eps=1e-06, n_num=200, ns_max=100000, ns_min=10000, n_tries=5, errorbar=0, verbose=0, verbose_IRLS=0)
#--- 3554.2674629688263 seconds ---
# ns = np.linspace(ns_min, ns_max, n_num)

python experiment.py -m 1 -K 3 -r 40 -np 8 -nsmax 100000 -nsmin 10000 -ntries 5 -nnum 200

python params_to_metric_SGaME.py -m 1 -K 3 -r 40 -np 8 -nsmax 100000 -nsmin 10000 -ntries 5 -nnum 200

python plotting_SGaME.py -m 1 -K 3 -r 40 -np 8 -nsmax 100000 -nsmin 10000 -ntries 5 -nnum 200 -errorbar 0

###
#Namespace(model=1, K=4, reps=40, nproc=8, maxit=2000, eps=1e-06, n_num=200, ns_max=100000, ns_min=10000, n_tries=5, errorbar=0, verbose=0, verbose_IRLS=0)
#--- 5609.496568202972 seconds ---
# ns = np.linspace(ns_min, ns_max, n_num)

python experiment.py -m 1 -K 4 -r 40 -np 8 -nsmax 100000 -nsmin 10000 -ntries 5 -nnum 200

python params_to_metric_SGaME.py -m 1 -K 4 -r 40 -np 8 -nsmax 100000 -nsmin 10000 -ntries 5 -nnum 200

python plotting_SGaME.py -m 1 -K 4 -r 40 -np 8 -nsmax 100000 -nsmin 10000 -ntries 5 -nnum 200 -errorbar 0

