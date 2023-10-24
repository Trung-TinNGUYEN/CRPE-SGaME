import numpy as np
import sys

rbar = [0,1,4,6] # cf. Lemma 1 in main text.

# Exact-fitted Settings
def loss_SGaME_exact(betas, As, Sigmas, beta, A, Sigma):
    
    # ####For debugging purposes
    # it = 0
    # jt = 0
    # beta = beta_list[it,jt,:,:]
    # A = A_list[it,jt,:,:]
    # Sigma = Sigma_list[it,jt,:,:]
    
    # ## Model 3. Hope with these seperation clusters.
    # betas   = np.array([[-7,0], [15,0]]) # [(d+1)xK]
    # As      = np.array([[2, 2], [-15,1]]) # [(d+1)xK]
    # Sigmas  = np.array([[0.3, 0.4]])**2 # [1xK]
    # print('beta = \n', beta, '\n betas = \n', betas, \
    #       '\n A = \n', A, '\n As = \n' , As, '\n Sigma = \n', Sigma,  '\n Sigmas = \n', Sigmas)
    ## Results:
    # beta = 
    #  [[-7.00032931 -7.00086037  0.        ]
    #  [14.99967069 14.99913963  0.        ]] 
    #  betas = 
    #  [[-7  0]
    #  [15  0]] 
    #  A = 
    #  [[  1.94282045   2.20693594   2.00933962]
    #  [-14.75452282 -15.49418917   0.86850571]] 
    #  As = 
    #  [[  2   2]
    #  [-15   1]] 
    #  Sigma = 
    #  [[0.0634427  0.03349318 0.13712274]] 
    #  Sigmas = 
    #  [[0.09 0.16]]
    
    Ks = betas.shape[1] # Ks =2
    K  = beta.shape[1]  # K = 3
    
    D = np.empty((K,Ks))

    for i in range(K):
        for j in range(Ks):
            # ## Testing
            # i = 0
            # j= 0
            # print('beta = \n', beta[:,j], '\n betas = \n', betas[:,j], \
            #       '\n A = \n', A[:,i], '\n As = \n' , As[:,i], '\n Sigma = \n', Sigma[:,i],  '\n Sigmas = \n', Sigmas[:,j])
            ## Results:
            # beta = 
            #  [-7.00032931 14.99967069] 
            #  betas = 
            #  [-7 15] 
            #  A = 
            #  [  1.94282045 -14.75452282] 
            #  As = 
            #  [  2 -15] 
            #  Sigma = 
            #  [0.0634427] 
            #  Sigmas = 
            #  [0.09]    
            D[i,j] = gauss_dist_SGaME(betas[:,j], As[:,j], Sigmas[:,j],\
                                      beta[:,i], A[:,i], Sigma[:,i])
            ## Results:
            #D[i,j]
            #Out[391]: 0.329543339434129
    # ### Result for D
    # array([[ 0.32954334,  2.20693594],
    #        [ 2.00933962, 14.75452282],
    #        [15.49418917,  0.86850571]])
    
    vor=[]
    for i in range(K):
        for j in range(Ks):
            if D[i,j] == np.min(D[i,:]):
                vor.append(j)
    # ## Results:
    # vor
    # Out[405]: [0, 0, 1]
    
    unique, counts = np.unique(vor, return_counts=True)
    # print(unique)
    # print(counts)
    summ = 0.0
    if (Ks==K):
        for i in range(K):
            # # ## Testing
            # i = 0
            # j= 0
            j = vor[i]
            if counts[vor[i]] == 1:
                summ += np.exp(beta[0,i]) * D[i,vor[i]]
    else:
        for i in range(K):
            j = vor[i]
            if counts[vor[i]] == 1:
                summ += np.exp(beta[0,i]) * D[i,vor[i]] 
                
            else:
                # ## Testing.
                ## To do: modified beta1_dist--Sigma_dist
                # i = 0
                j = vor[i]
                rb = rbar[counts[j]]
                beta1_dist = (np.linalg.norm(beta[1,i]-betas[1,j]))**rb  ## Previous error here! betas[1,j] not betas[1,:]?
                b_dist = (np.linalg.norm(A[0,i] - As[0,j]))**rb
                # print(beta1_dist + b_dist)
                A_dist = (np.linalg.norm(A[1,i] - As[1,j]))**(rb/2.0)
                Sigma_dist = (np.linalg.norm(Sigma[:,i] - Sigmas[:,j]))**(rb/2.0)
                
                summ += np.exp(beta[0,i]) * (beta1_dist + b_dist + A_dist + Sigma_dist)

    for j in range(Ks):
        exp_beta0 = 0

        for i in range(K):
            if vor[i] == j:
                exp_beta0 += np.exp(beta[0,i])

        summ += np.abs(exp_beta0 - np.exp(betas[0,j]))

    return summ

def gauss_dist_SGaME(betas, As, Sigmas, beta, A, Sigma):
    return np.linalg.norm(betas[1] - beta[1]) + np.linalg.norm(As[0]-A[0]) \
            + np.linalg.norm(As[1]-A[1]) + np.linalg.norm(Sigmas-Sigma)


