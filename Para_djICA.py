#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 21:48:44 2018

@author: aaco
"""


import numpy as np

from scipy import signal

from sklearn.decomposition import FastICA, PCA

#%% 
################### helper functions ###################

def mySigmoid(X):
    tmp = 1 + np.exp(-X)
    return np.divide(1, tmp)

########################################################


#%%
def message_builder(X):
    
    d, n = X.shape
    
    num_ind_compo = 3 
    num_sites = 2
    num_samples_site = np.int(n / num_sites)
    st_id = 0
    en_id = num_samples_site
    
    
    
    C = (1.0 /n) * np.dot(X, X.T)
    U, S, V = np.linalg.svd(C)
    Uk = U[:, :num_ind_compo]
    
    args=dict()
    dataX = dict()
    
    # build message and initialize 
    
    for s in range(num_sites):
        Xs = np.array(X[:, st_id : en_id])
        st_id += num_samples_site
        en_id += num_samples_site
        args[s] =  {'U' : Uk , 'W': np.eye(num_ind_compo), 
            'b': np.zeros([num_ind_compo, 1]), 'rho': 0.015 / np.log(5), 
            'iter': 1}
        dataX[s] = {'X': Xs}
    
    
    return args, dataX

########################################################
#%%
def local_ica(args,dataX):    
  
## data structure 
## arg= { 'rawdata': {'X': X, 'U':U} , 'W':W, 'b':b, 'rho': rho, 'iter: iteration'}     

          
    X = np.array(dataX['X'])
    D, N = X.shape
    block =1# (N/20) ** (0.5)
    ones = np.ones([N, 1])
    
    # project data on the reduced subspace
    U = np.array(args['U'])
    Xred = np.dot(U.T, X)
    BI = block * np.eye(Xred.shape[0])
    
    # extract the data from the remote
    W = np.array(args['W'])
    b = np.array(args['b'])
    rho = np.array(args['rho'])
    itr = np.array(args['iter'])
    
    # take grdient step
    Z = np.dot(W, Xred) + np.dot(b, ones.T)
    Y = mySigmoid(Z)
    G = rho * np.dot(BI + np.dot(1 - 2*Y, Z.T), W)
    h = rho * np.sum(1 - 2*Y, axis = 1)
    h = h.reshape([Xred.shape[0], 1])

    #rho = rho/(2*itr)
    
    # dumping the computation to be sent to the remote 
    computationOutput = {'U': U.tolist(), 'G' : G.tolist(), 'h' : h.tolist(),
                         'rho' : rho, 'W' : W.tolist(), 'b' : b.tolist(),
                         'iter': args['iter']}
    
    return computationOutput
   
#%% 
    
def remote_ica(args):
    # args = {1:{ outputOfLocal1 }, 2:{}, 3:{},...}
    # outputOfLocal1 = {'X'= X, 'U'= U, 'G' : G, 'h' : h, 'rho' : rho, 'W' : W, 'b' : b, 'iter': iter}
   
    
    n_site = len(args)
    
    thr = 1e8
    maxItr = 3000
    
    # repeat the initial values 
    
    # recover the previous iteration values from what the sites send
    
    itr = args[0]['iter']
    W = np.array(args[0]['W'])
    b = np.array(args[0]['b'])
    U = np.array(args[0]['U'])
    rho = args[0]['rho']
    
    K = len(W)   # num indpendent components num_ind_compo
    
    gradSum = 0.0
    biasGradSum = 0.0    
    
    
    for i in range(0, n_site):
        gradSum += np.array(args[i]['G'])
        biasGradSum += np.array(args[i]['h'])
    
    # printout 
    if itr % 100 == 0: 
        print('Grad norm:', np.linalg.norm(gradSum))
        print('Iter:', itr)
        
    
    computationOutput = {}
    if itr < maxItr and np.linalg.norm(gradSum) > 1e-8:
    #    sys.stderr.write("\n Updating W")    
        W = np.add(W, gradSum, casting='unsafe')
    #    sys.stderr.write("\n Updating b")    
        b = np.add(b, biasGradSum, casting='unsafe')
        itr += 1
    
    #      check blowout and update rho if needed
        if np.max(np.abs(W)) >= thr:
    #      sys.stderr.write("\n Blowout detected. Restarting...")    
            rho = rho * 0.8
    #      initialize W and b again
            #W = np.eye(K)
            #W = np.random.uniform(-1,1,(K,K))
            W = np.random.normal(0,0.5,(K,K))
            #b = np.zeros([K, 1])
            b = np.random.normal(0,1,(K,1))
            itr = 1
            
     
        # send these values to local sites
        
        for icc in range(n_site):
           # X = np.array(args[icc]['X'])
            computationOutput[icc] = {'U': U.tolist(), 'W' : W.tolist(), 'b' : b.tolist(), 'rho' : rho, 'iter' : itr }
    
            
        
    else:
        itr += 1
        #res_file = 'mixing_matrix.npz'
        #A = np.load(res_file)['arr_0']
        Ahat = np.linalg.pinv(np.dot(W, U.T))
        #err = np.linalg.norm(A - Ahat, 'fro')
        # sys.stderr.write("Done with djICA! Error : {}".format(err)+"\n")
        # send these values to local sites
        computationOutput[0] = {'status': '1', 'W' : W.tolist(), 'U' : U.tolist(),
                         'iter' : itr, 'mixingMatrix': Ahat.tolist()}
        
    # send results
    return computationOutput      



#%% Performance metric based on Amari paper
def Amari_ISI(P,L):

    slak_1 = 0
    slak_2 = 0
    slak_1a = 0
    slak_2a = 0

    for i in range(L):
        for j in range(L):
           slak_1 = slak_1 + ((np.abs(P[i,j])/np.max(np.abs(P[i,:]))))
        
        slak_1a = slak_1a + slak_1 - 1
        slak_1 = 0
    
    
    for j in range(L):
        for i in range(L):
           slak_2 = slak_2 + ((np.abs(P[i,j])/np.max(np.abs(P[:,j]))))
        
        slak_2a = slak_2a + slak_2 - 1
        slak_2 = 0
    
    
    return  (1/(2*L*(L - 1)))*(slak_1a + slak_2a)


#%% master function
def serial_simulator(args, dataX, A):
    # args structure
    # args = {{1: dataOfSite1, 2: dataOfSite2...}}
    # dataOfSite1 = {'X': X, 'U':U , 'W':W, 'b':b, 'rho': rho, 'iter': iter}}
    # local_computation = {1:{ outputOfLocal1 }, 2:{}, 3:{},...}
    # outputOfLocal1 = {'X'= X, 'U'= U, 'G' : G, 'h' : h, 'rho' : rho, 'W' : W, 'b' : b, 'iter': iter}
    
    K = len(args[0]['W'])  # num_ind_compo
    message = args
    
    n_sites = len(message)
    local_computation = {}
    
    while message[0]['iter'] <= 3000 and 'status' not in message[0]:  
        # one round of serial locals
        for i in range(n_sites): 
            local_computation[i] = local_ica(message[i],dataX[i])
            
        # all sent to remote   
        message = remote_ica(local_computation) 
    
    # estimate of the mixing matrix
    A_est = np.array(message[0]['mixingMatrix'])  
    
    # preparing for assessing the A_est 
    Amari = np.dot(np.dot(np.array(message[0]['W']), np.array(message[0]['U']).T), A)
    
    print('Amari ISI djICA:', Amari_ISI(Amari,K))    
    return A_est

#%% Using sklearn ICA centralized 
# Compute ICA
def C_Fast_ICA(X, A, num_ind_compo):
    
    X = X.T
    
    ica = FastICA(n_components=num_ind_compo)
    S_ = ica.fit_transform(X)  # Reconstruct signals
    A_ = ica.mixing_  # Get estimated mixing matrix
    
    # We can `prove` that the ICA model applies by reverting the unmixing.
    assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)
    
    # Amari ISI
    P = np.dot(A,np.linalg.pinv(A_))

    print('Amari ISI Centralized fastICA:', Amari_ISI(P,num_ind_compo))
#%%
# Generate data X and mixing matrix A    
X, A, n_ind_compo = generate_synthetic(2)

# build and initialize the messages 
args, dataX = message_builder(X)  

# Run the messages through local and remote engines
A_est = serial_simulator(args, dataX, A)

C_Fast_ICA(X, A, n_ind_compo)