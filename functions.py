import numpy as np
from copy import deepcopy
from scipy.stats import multivariate_normal
from numpy.linalg import inv
import sys
from scipy.special import logsumexp


### Some further functions
def sample_SGaMEs(n, seed = 0):
    """ Sample from the GLLiM. """
    beta, A, Sigma = get_params(n)
    
    return sample_SGaME(beta, A, Sigma, n)

def init_params_SGaME(n, K):
    """ Starting values for EM algorithm. """
    (betas, As, Sigmas) = get_params(n)

    beta_start     = np.empty([d+1,K-1])
    A_start        = np.empty([d+1, K])
    Sigma_start    = np.empty([1, K])
    
    inds = range(Ks)

    # Make a partition of starting values near the true components.
    while True:
        s_inds = np.random.choice(inds, size=K)
        unique,counts = np.unique(s_inds, return_counts=True)

        if unique.size==Ks:
            break
    
    for i in range(K):
        if i < (K-1):
            beta_start[i,:]    = betas[:,s_inds[i]] + np.random.normal(0, 0.005*n**(-0.083), size=(d+1,1))
        A_start[i,:]     = As[:,s_inds[i]] + np.random.normal(0, 0.005*n**(-0.083), size=(d+1,1))
        Sigma_start[i] = Sigmas[:,s_inds[i]] + np.abs(np.random.normal(0, 0.0005*n**(-0.25), size=1))
        
    return (beta_start, A_start, Sigma_start)

def sample_SGaME(betas, As, Sigmas, nb_data, seed = 0):
    """
    Draw nb_data samples (Xi, Yi), i = 1,...,nb_data, from a supervised Gaussian locally-linear mapping (GLLiM).

    Parameters
    ----------
    betas : ([nb_mixture, dim_X +1], np.array)
        Intercept and coefficience vectors for weights.
    As : ([nb_mixture, dim_X +1] np.array)
        Intercept vector and Regressor matrix of location parameters of normal distribution.            
    Sigmas : ([nb_mixture, dim_data_Y, dim_data_Y] np.array)
        Scale parameters (Gaussian experts' covariance matrices) of normal distribution.            
    nb_data : int
        Sample size.    
    seed : int
        Starting number for the random number generator.    
                    
    Returns
    -------
    data_X : ([nb_data, dim_data_X+1] np.array) dim_data_X = dim_X+1 with the first column := 1.
        Input random sample.
    data_Y : ([nb_data] np.array)
        Output random sample.

    """
    ######################################################################################## 
    # Sample the input data: data_X.
    ########################################################################################   
    
    # ## Quick test
    # seed = 0
    # nb_data = 4
    
    
    rng = np.random.default_rng(seed)
    
    # Draw nb_data data_X samples from uniform distribution on [0,1].
    
    data_Xd = rng.uniform(low=0.0, high=1.0, size=nb_data).reshape(nb_data, betas.shape[1]-1)
    data_X = np.concatenate((np.ones((nb_data,1)),data_Xd), axis = 1) # ([nb_data, dim_data_X+1] np.array)
    
    ########################################################################################
    # Sample the output data: data_Y via latent_Z variable.
    ########################################################################################
    # dim_data_Y = As.shape[1]
    # dim_data_Y = l
    data_Y = np.zeros((nb_data))

    # Calculate the softmax gating network probabilites

    gating_prob = sample_softmax_gate(data_X, betas)
    
    # ## Test whether the sum of the rows in gating_prob is equal to 1 in each column.
    # ## print(gating_prob)
    # print('Sum of the rows in gating_prob is equal to 1 = ', \
    #       np.all(np.sum(gating_prob, axis = 1).reshape((nb_data,1))-np.ones((nb_data,1))<1e-16))
    
    nb_mixture = betas.shape[0]
    latent_Z =  np.zeros((nb_data, nb_mixture))
    kclass_Y =  np.zeros((nb_data, 1))
    
    for n in range(nb_data):
        Znk = rng.multinomial(1, gating_prob[n], size = 1)[0]
        latent_Z[n] = Znk
        zn = np.where(Znk == 1)[0]
        kclass_Y[n] = zn[0]
        # Sample Y
        # data_Y[n, None] = rng.multivariate_normal(mean = (As[zn[0], :, :]@data_X[n, None].T).reshape(dim_data_Y),
        #                                               cov = Sigmas[zn[0], :, :])
        data_Y[n] = rng.multivariate_normal(mean = (As[zn[0], :]@data_X[n, None].T),
                                                      cov = Sigmas[zn[0]].reshape(1,1))
        
    return (data_X, data_Y)

def sample_softmax_gate(data_X, betas):
    """
    Calculate the softmax gating network probabilites.

    Parameters
    ----------
    data_X: ([nb_data, dim_X + 1] np.array)
        Input sample. dim_data_X = dim_X+1 with the first column := 1.    
    betas : ([nb_mixture, dim_data_X +1], np.array)
        Intercept and coefficience vectors for weights.
        
    Returns
    -------
    gating_prob: ([nb_data, nb_mixture], np.array)
        Softmax gating network probabilites.
    """
    ## Test.
    # data_X = X
    
    (nb_data, dim_data) = np.shape(data_X)
    nb_mixture = betas.shape[0]
    gating_prob_sample = np.zeros((nb_data, nb_mixture))
    
    mat_Xbeta = np.exp(data_X@(betas.T))
    rowSum_mat_Xbeta = np.sum(mat_Xbeta, axis = 1).reshape(nb_data,1)
    sum_mat_Xbeta = np.tile(rowSum_mat_Xbeta, reps = (1,nb_mixture))
    
    ## Test
    # print(sum_mat_Xbeta)   
    # print(sum_mat_Xbeta.shape)
    
    # Compute softmax_gate(X, beta) with beta[-1,:] = 0
    gating_prob_sample = mat_Xbeta/sum_mat_Xbeta
    
    return gating_prob_sample

def softmax_gate(X, beta):
    """
    Calculate the softmax gating network probabilites.

    Parameters
    ----------
    X: ([nb_data, dim_X + 1] np.array)
        Input sample with dim_data_X = dim_X+1 with the first column := 1.    
    betas : ([nb_mixture-1, dim_data_X +1], np.array)
        Intercept and coefficience vectors for weights.
        
    Returns
    -------
    gating_prob: ([nb_data, nb_mixture], np.array)
        Softmax gating network probabilites.
    """
    ## Test.
    # beta = beta_start
    
    (nb_data, dim_data_X) = np.shape(X)
    nb_mixture = beta.shape[0] + 1
    betaK = np.concatenate((beta,np.zeros((1,dim_data_X))))
    gating_prob = np.zeros((nb_data, nb_mixture))
    
    mat_Xbeta = np.exp(X@(betaK.T))
    rowSum_mat_Xbeta = np.sum(mat_Xbeta, axis = 1).reshape(nb_data,1)
    sum_mat_Xbeta = np.tile(rowSum_mat_Xbeta, reps = (1,nb_mixture))
    
    ## Test
    # print(sum_mat_Xbeta)   
    # print(sum_mat_Xbeta.shape)
    
    # Compute softmax_gate(X, beta) with beta[-1,:] = 0
    gating_prob = mat_Xbeta/sum_mat_Xbeta
    
    # ## Test whether the sum of the rows in gating_prob is equal to 1 in each column.
    # ## print(gating_prob)
    # print('Sum of the rows in gating_prob is equal to 1 = ', \
    #       np.all(np.sum(gating_prob, axis = 1).reshape((nb_data,1))-np.ones((nb_data,1))<1e-16))
    
    return gating_prob


def log_LL(X, Y, beta, A, Sigma):
    # # # Test
    # beta = beta_start
    # Sigma = Sigma_start
    # A = A_start
    
    # LL = np.ndarray((max_iter,1))
    n = X.shape[0]
    K = beta.shape[1] + 1
    
    pik = softmax_gate(X, beta)
    logLL = 0
    mu = X@A.T # [n*(d+1)x(K*(d+1)).T = nxK]
    tau =  np.ndarray((n,K))
    
    for i in range(n):
        for k in range(K):
            tau[i,k] = pik[i,k]*multivariate_normal.pdf(Y[i], mu[i,k], Sigma[k].reshape(1,1))
    
    # print(tau)
    logLL = np.sum(np.log(np.sum(tau, axis = 1)))
    
    return logLL

def e_SGaME(X, Y, beta, A, Sigma):
    # # # Test
    # beta = beta_start
    # Sigma = Sigma_start
    # A = A_start
    
    # LL = np.ndarray((max_iter,1))
    n = X.shape[0]
    K = beta.shape[0] + 1
    
    pik = softmax_gate(X, beta)
    mu = X@A.T # [n*(d+1)x(K*(d+1)).T = nxK]
    tau =  np.ndarray((n,K))
    logLL = 0
    
    # print('K = ', K)
    
    for i in range(n):
        for k in range(K):
            # print('kbefore = ', k)
            tau[i,k] = pik[i,k]*multivariate_normal.pdf(Y[i], mu[i,k], Sigma[k].reshape(1,1))
            # print('k = ', k, 'pik.shape', pik.shape, 'mu.shape', mu.shape)
    
    # print(tau)
    ## Compute log-likelihood.
    logLL = np.sum(np.log(np.sum(tau, axis = 1)))
    
    ## Update Zik
    tau = tau/(np.sum(tau, axis = 1).reshape(n,1))
 
    
    return (logLL, tau)


def Obj(X, Y, tau, u, Gammak):
    #Compute the value of the objective function
    
    Val_Obj = (tau.reshape(1,X.shape[0]))@((X@u.reshape(X.shape[1],1) \
                                            - Y.reshape(X.shape[0],1))**2) #[1x1]
    
    if Gammak==0:
        Val_Obj = Val_Obj/2    
    
    return Val_Obj

def SoTh(numerator, Gammak):
    if Gammak==0:
        return numerator
    elif numerator>Gammak:
        return numerator-Gammak
    elif numerator<-Gammak:
        return numerator+Gammak
    else:
        return 0


# Q(pi;.) function in M-step

def Fs(X, tau, beta):
    
    # n = X.shape[0]
    # K = beta.shape[1] + 1
    
    pik = softmax_gate(X, beta)
    
    Qw = np.sum(tau*np.log(pik))
    
    return Qw

### Find MINIMUM of penalized function
## 3.3.1. Expert network with Gaussian outputs in Gpm.step().
## (20)+(21)_(23-24-25): coordinate descent for maximizing L_Qk in CoorGateP1().
# u vector of parameters (beta0, beta1) represent beta.
# u vector of parameters (b, A) represent A.
# true tau is in c_k.
# tau vector represent by d_k.

def CoorLQk(X, Y, tau, u, Gammak=0):
    
    esp_CoorLQk = 1e-6 ## Stopping condition
    d = X.shape[1]
    d1 = d - 1
    n = X.shape[0]
    
    u = u.reshape(1,d)
    Val = Obj(X, Y, tau, u, Gammak) # (27): Y:=c_k, tau:=d_k, u:=beta_new[k,:]
    Xmat = np.delete(X, 0, axis = 1) #[nxd1]
    Sum = (tau.reshape(1,n))@(Xmat**2) #[1xn nxd1 = 1xd1]
    # print('Shape(Sum) = ', Sum.shape)
    TauU = np.sum(tau) #[1x1]
    iiter_CoorLQk = 0
    
    while True:
        Val1 = Val
        beta0 = u[:,0]
        beta1 = np.delete(u, 0, axis = 1).reshape(1, d1)
        # print(beta0)
        # print(beta1.shape)
        
        for j in range(d1):
            # [nx1 - nxd @ dx1 + 1x(nx1) = nx1].
            rij = Y.reshape(n,1) - X@(u.reshape(X.shape[1],1)) \
                + (beta1[j]*Xmat[:,j]).reshape(n,1)
            # [ nx1* nx1 * nx1 = nx1--> np.sum() = 1x1]    
            numerator = np.sum(rij*(tau.reshape(n,1))*(Xmat[:,j].reshape(n,1)), axis = 0)
            # print('rij =', rij)
            # print('shape(rij) =', rij.shape)
            # [nx1 * nx1]
            denominator = np.sum((tau.reshape(n,1))*(Xmat[:,j].reshape(n,1))**2,axis = 0)
            # beta1[j] = SoTh(numerator, Gammak)/denominator # (28) or (21)
            beta1[j]   = numerator/denominator # (28) or (21) No penalization!
            
            # [1xn @(nx1 - nxd1 @ d1x1) = 1x1]
            beta0 = (tau.reshape(1,n))@(Y.reshape(n,1) - Xmat@(beta1.reshape(d1,1)))
            beta0 = beta0/TauU # (29) or (22) [1x1]
            
            u = np.concatenate((beta0,beta1.reshape(1,d1)), axis = 1) #[1x(d1+1)]
        
        u = u.reshape(1,d)
        Val = Obj(X, Y, tau, u, Gammak)
        
        # Stopping criterion.
        if ((Val1 - Val) < esp_CoorLQk):
            # print('Val1 - Val = ', Val1 - Val)
            # print('Val1 - Val < esp_CoorLQk? ', (Val1 - Val) < esp_CoorLQk)
            # print(np.all((Val1 - Val) < esp_CoorLQk))
            break
        iiter_CoorLQk += 1
        # print('iiter_CoorLQk = ', iiter_CoorLQk)
    # return Val
    return (u, iiter_CoorLQk)
    # return (u,rij,numerator,denominator,beta1,beta0)

            
def mbeta_SGaME(X, Y, tau, beta, max_iter_mbeta = 1000):
    
    # LL = np.ndarray((max_iter,1))
    n = X.shape[0]
    K = tau.shape[1]
    
    P_k = np.zeros(n)
    d_k = 1/4*np.ones(n)
    c_k = np.zeros(n)
    
    Stepsize = 0.5
    esp_Q = 1e-5 #threshold for Q value
    beta_new    = beta 
    
    for iiter_mbeta in range(max_iter_mbeta):
        
        beta_old = beta_new
        Q_old = Fs(X, tau, beta_old)
        # print('Q_old = ', Q_old)
        
        for k in range(K-1):
            # #First: compute the quadratic approximation w.r.t (w_k): L_Qk
            # value of Pi_k
            P = softmax_gate(X, beta_old)
            P_k = P[:,k]
            
            c_k = X@(beta_new[k,:].reshape(beta_new.shape[1],1)) + 4*(tau[:,k]-P_k).reshape(n,1)
            
            # #Second: coordinate descent for maximizing L_Qk
            beta_new[k,:] = CoorLQk(X, c_k, d_k, beta_new[k,:])[0]
        
        Q_new = Fs(X, tau, beta_new)
        # print('Q_new = ', Q_new)
        
        # Backtracking line search.
        t = 1 
        while (Q_new < Q_old):
            t = t*Stepsize
            beta_new = beta_new*t + beta_old*(1-t)
            Q_new = Fs(X, tau, beta_new)
            # print('Q_new_line = ', Q_new)
            
        # print('Q_new = ', Q_new)    
        
        if ((Q_new - Q_old) < esp_Q):
            print('mbeta_SGaME converged after iterations = ', iiter_mbeta)
            break
    
    # print('Ok for mbeta_SGaME!')
    
    return beta_new

def mAb_SGaME(X, Y, tau, A, Sigma):
    
    n = X.shape[0]
    K = tau.shape[1]
    # A1 = A
    A1 = np.zeros((K,X.shape[1]))
    for k in range(K):
        # print('k = ', k)
        (A1[k,:], iiter_mAb_SGaME) = CoorLQk(X, Y, tau[:,k], A[k,:])
        print('mAb_SGaME converged after iterations = ', iiter_mAb_SGaME)
        
    return A1

def mSigma_SGaME(X, Y, tau, A, Sigma):
    
    n = X.shape[0]
    K = tau.shape[1]
    S1 = 0
    
    for k in range(K):
        temp = (tau[:,k].reshape(1,n))@((Y.reshape(n,1) - X@(A[k,:].reshape(X.shape[1],1)))**2)
        Sigma[k] = temp/np.sum(tau[:,k]) 
        
    return Sigma

def em_SGaME(X, Y, beta_start, A_start, Sigma_start, max_iter=1000, eps=1e-6):

    beta_prev   = beta_start    #([nb_mixture, dim_data_X +1], np.array)
    A_prev      = A_start       #([nb_mixture, dim_data_X +1] np.array)
    Sigma_prev  = Sigma_start   #([nb_mixture] np.array)
 
    beta_new    = beta_start    
    A_new       = A_start
    Sigma_new   = Sigma_start
    
    K = A_start.shape[0]
    n = X.shape[0]
    d = X.shape[1] # d := d + 1
    # l = Y.shape[1]
    LL= np.empty((max_iter,1))
    converged = False
    
    for iiter in range(max_iter):
        
        # Test
        # iiter = 0
        
        ## Compute log-likelihood and E-Z step.
        # print("E-Z") 
        (LL[iiter,0], tau) = e_SGaME(X, Y, beta_start, A_start, Sigma_start)
        # LL[iiter,0] = log_LL(X, Y, beta_start, A_start, Sigma_start)
        
        ##  M-step:
        # print("M")
        
        # Update A_new.
        A_new = mAb_SGaME(X, Y, tau, A_start, Sigma_start)
        
        
        # Update beta_new.
        beta_new = mbeta_SGaME(X, Y, tau, beta_start)
        
        
        ## Re-run the E-step.
        
        tau = e_SGaME(X, Y, beta_new, A_new, Sigma_start)[1]
        
        Sigma_new = mSigma_SGaME(X, Y, tau, A_new, Sigma_start)
        
        # Re-compute log-likehood.
        
        LL[iiter,0] = e_SGaME(X, Y, beta_new, A_new, Sigma_new)[0]
        
        # Update Sigma_new.
        # Update sigma_k^2 according to the standard update of a weighted Gaussian regression
        
        print('iiter = ', iiter)
        ## Stoping criterion.
        if (iiter >= 3):
            deltaLL_total = np.amax(LL[0:iiter,0])-np.amin(LL[0:iiter,0])
            deltaLL = LL[iiter,0]-LL[iiter-1,0]  # EVERY EM Iteration THIS MUST INCREASE
            converged = bool(deltaLL <= eps*deltaLL_total) 
            
        print('converged = ', converged)
        
        # if max(np.linalg.norm(beta_new - beta_prev), np.linalg.norm(A_new - A_prev),\
        #     np.linalg.norm(Sigma_new - Sigma_prev))< eps or (iiter > max_iter) or converged:
        #     break
        if max(np.linalg.norm(beta_new - beta_prev), np.linalg.norm(A_new - A_prev),\
            np.linalg.norm(Sigma_new - Sigma_prev))< eps or (iiter > max_iter):
            break    
    
        beta_prev    = deepcopy(beta_new)
        A_prev     = deepcopy(A_new)
        Sigma_prev = deepcopy(Sigma_new)
        
    return (beta_prev, A_prev, Sigma_prev, iiter, LL)

### Additional functions
def posterior_Gaussian_gate(data, weight, mean, cov):
    """
    Compute responsibilities in a Gaussian Mixture Model.

    Parameters
    ----------
    data : ([nb_data, dim_data] np.array)
        Input sample.    
    weight : ([nb_mixture], np.array)
        Mixing proportion.
    mean : ([nb_mixture, dim_data], np.array)
        Means of Gaussian components.
    cov : ([nb_mixture, dim_data, dim_data], np.array)
        Covariance matrices of Gaussian components.
        
    Returns
    -------
    respons: ([nb_data, nb_mixture], np.array)
        Responsibilities.
    log_pik_Nik : ([nb_data, nb_mixture], np.array)
          Log of product between the weights and the PDF of a multivariate GMM.  
    """
    # loglik : np.float32
    #     Log-likelihood of GMM.    
    
    nb_mixture = len(weight)
    log_pik_Nik = Gaussian_weight_pdf(data, weight, mean, cov)[0]
    log_sum_exp_pik_Nik = logsumexp_SGaME(log_pik_Nik, 1)
    log_responsik = log_pik_Nik - log_sum_exp_pik_Nik@np.ones((1, nb_mixture))
    respons = np.exp(log_responsik)
    # loglik = np.sum(log_sum_exp_pik_Nik)
    
    return respons, log_pik_Nik

def Gaussian_weight_pdf(data, weight, mean, cov):
    """
    Calculate the log product between the weights and the PDF 
    of a Gaussian distribution.

    Parameters
    ----------
    data : ([nb_data, dim_data] np.array)
        Input sample.    
    weight : ([nb_mixture], np.array)
        Mixing proportion.
    mean : ([nb_mixture, dim_data], np.array)
        Means of Gaussian components.
    cov : ([nb_mixture, dim_data, dim_data], np.array)
        Covariance matrices of Gaussian components.
    Returns
    -------
    log_Nik : ([nb_data, nb_mixture], np.array)
        Log Pdf of a Gaussian distribution PDF.
        
    log_pik_Nik : ([nb_data, nb_mixture], np.array)
        Log of product between the weights 
        and the PDF of a Gaussian distribution.  
    """
    
    (nb_data, dim_data) = np.shape(data)
    nb_mixture = len(weight)
    log_Nik = np.zeros((nb_data, nb_mixture))
    log_pik_Nik = np.zeros((nb_data, nb_mixture))
    
    for k in range(nb_mixture):
        log_Nik[:, k, None] = Gaussian_ndata(data, mean[None,k,:], cov[k])[0]
        log_pik_Nik[:, k, None] = np.ones((nb_data, 1))* np.log(weight[k]) + log_Nik[:, k, None]
    
    return log_pik_Nik, log_Nik

def Gaussian_ndata(data, mean, cov):
    """
    Calculate pdf of a Gaussian distribution over nb_data.

    Parameters
    ----------
    data : ([nb_data, dim_data] np.array)
        Input sample.
    mean : ([1, dim_data], np.array)
        Means of Gaussian.
    cov : ([dim_data, dim_data], np.array)
        Covariance matrices of Gaussian.
        
    Returns
    -------
    log_Ni : ([nb_data, 1], np.array)
        Log pdf of a Gaussian distribution.
    Ni : ([nb_data, 1], np.array)
        Pdf of a Gaussian distribution.
    """
    
    # ## Code from scratch  without using multivariate_normal.
    # (nb_data, dim_data) = np.shape(data)
    # det_cov = np.linalg.inv(cov)
    # z = ((data - np.ones((nb_data, 1))@mean)@det_cov)*(data - np.ones((nb_data, 1))@mean)
    # mahalanobis = np.sum(z, axis=1, keepdims=True)
    # log_Ni = -(dim_data/2)*np.log(2*np.pi) - 0.5*np.log(det_cov) - 0.5*mahalanobis
    # Ni = np.exp(log_Ni)
    
    (nb_data, dim_data) = np.shape(data)
    log_Ni = np.zeros((nb_data, 1))
    Ni = np.ones((nb_data, 1))
    
    for n in range(nb_data):
        log_Ni[n, :, None] =\
            multivariate_normal.logpdf(data[n, :, None], mean,
                                       cov+sys.float_info.epsilon*np.diag(np.ones(dim_data)))
        Ni[n, :, None] =\
            multivariate_normal.pdf(data[n, :, None], mean,
                                    cov+cov+sys.float_info.epsilon*np.diag(np.ones(dim_data)))
    
    return log_Ni, Ni

def logsumexp_SGaME(x, dimension):
    """
    Compute log(sum(exp(x),dim)) while avoiding numerical underflow.
    By default dim = 1 (row).

    Parameters
    ----------
    x : np.array
        Input data.
    dimension : int
        0: Column sum 1: Row sum.

    Returns
    -------
    log_sum_exp: np.float64
        Value of log(sum(exp(x),dim)).

    """
    # Subtract the largest value in each row
    x_max = np.amax(x, dimension, keepdims=True)
    x = x - x_max
    x_log_sum_exp = x_max + np.log(np.sum(np.exp(x), dimension, keepdims=True))
    
    x_max_check_inf = np.isinf(x_max)
    if np.sum(x_max_check_inf) > 0:
        x_log_sum_exp[x_max_check_inf] = x_max[x_max_check_inf]
    
    return x_log_sum_exp

## log_gauss_densities.py
_LOG_2PI = np.log(2 * np.pi)

# log of pdf for gaussian distributuion with diagonal covariance matrix
def loggausspdf(X, mu, cov):
    if len(X.shape)==1:
        D=1
    else:
        D = X.shape[1]
    
    logDetCov = D*np.log(cov)
    dxM = X - mu
    L = np.sqrt(cov)
    xRinv = 1/L * dxM
    mahalaDx = np.sum(xRinv**2, axis=1)
    y = - 0.5 * (logDetCov + D*_LOG_2PI + mahalaDx)
    return y

def gausspdf(X, mu, cov):
    return np.exp(loggausspdf(X, mu, cov))

# log of pdf for gaussian distributuion with full covariance matrix (cholesky factorization for stability)
def chol_loggausspdf(X, mu, cov):

    D = X.shape[0]
    
    X = X - mu #DxN
    U = np.linalg.cholesky(cov + sys.float_info.epsilon).T #DxD
    Q = np.linalg.solve(U.T,X)
    q = np.sum(Q**2, axis=0)
    c = D*_LOG_2PI + 2*np.sum(np.log(np.diag(U)))
    y = -0.5 * (c + q)

    return y 