#!/usr/bin/env python
# Copyright (c) 2014 - 2018  Mateo Rojas-Carulla  [mrojascarulla@gmail.com]
# All rights reserved.  See the file COPYING for license terms.

import autograd.numpy as np_aut
import autograd

import numpy as np
import scipy as sc
from scipy import io
from scipy.spatial.distance import pdist, squareform
import sys
import time

from sklearn import linear_model
from sklearn import svm
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from matplotlib import pyplot as pl
from matplotlib import rc
import matplotlib as mpl

from sklearn.neighbors import KernelDensity
from sklearn.cross_validation import KFold
import os

import cPickle as pickle
import picos as pic
from cvxopt import matrix, solvers

def split_train_valid(x, y, n_ex, valid_split=0.1):
    n_ex_cum = np.append(0, np.cumsum(n_ex))
    n_ex_train, n_ex_valid = [], []
    train_x, train_y, valid_x, valid_y = [], [], [], []

    for i in range(len(n_ex)):
        n_train_task = int((1 - valid_split) * n_ex[i])
        train_x.append(x[n_ex_cum[i]:n_ex_cum[i] + n_train_task])
        train_y.append(y[n_ex_cum[i]:n_ex_cum[i] + n_train_task])

        valid_x.append(x[n_ex_cum[i] + n_train_task:n_ex_cum[i + 1]])
        valid_y.append(y[n_ex_cum[i] + n_train_task:n_ex_cum[i + 1]])
        
        n_ex_train.append(n_train_task)
        n_ex_valid.append(n_ex[i] - n_train_task)

    train_x = np.concatenate(train_x, 0)
    valid_x = np.concatenate(valid_x, 0)
    train_y = np.concatenate(train_y, 0)
    valid_y = np.concatenate(valid_y, 0)

    n_ex_train = np.array(n_ex_train)
    n_ex_valid = np.array(n_ex_valid)

    return train_x, train_y, valid_x, valid_y, n_ex_train, n_ex_valid



def np_getDistances(x,y):
    K = (x[:,:, np.newaxis] - y.T)
    return np.linalg.norm(K,axis = 1)

    
#Select top 11 predictors from Lasso
def lasso_alpha_search_synt(X,Y):

    exit_loop = False
    alpha_lasso = 0.2
    step = 0.02
    num_iters = 1000
    count = 0
    n = 11

    while(not exit_loop and count < num_iters):
            count = count + 1

            regr = linear_model.Lasso(alpha = alpha_lasso)
            regr.fit(X,Y.flatten())
            zeros =  np.where(np.abs(regr.coef_) < 0.00000000001)

            nonzeros = X.shape[1]-zeros[0].shape[0]

            if(nonzeros >= n and nonzeros<n+1):
                    exit_loop = True
            if nonzeros<n:
                    alpha_lasso -= step
            else:
                    step /= 2
                    alpha_lasso += step


    mask = np.ones(X.shape[1],dtype = bool)
    mask[zeros] = False
    genes = []
    index_mask = np.where(mask == True)[0]

    return mask



#Given a number of training tasks, the total number of examples and the number per task, return a boolean mask (for SMTL)
def mask_training_tasks(n_tasks,n_s,n_tot,n_pred):
    mask = np.zeros((n_tot,n_pred),dtype = bool)
    n_each = n_tot/n_tasks
    for t in range(n_tasks):
        mask[t*n_each:t*n_each+n_s,:] = True
    return mask


#-------------------------------------------------------------------------------
# Run and SDP to find a feasible solution when optimising (4)
#-----------------------------------------------------------------------------

def find_init_sol(Cov_ctr,s_size, n_size):

    shape = Cov_ctr.shape[0]
    sdp = pic.Problem()
    X = sdp.add_variable('X', (shape,shape), vtype='symmetric')
    Cov_ctr = pic.new_param('M', matrix(Cov_ctr))

    #Matrix has to be spd
    sdp.add_constraint(X>>0)
    for i in range(shape):
        for j in range(i+1):
            if i==shape-1 and j>=s_size and j<shape-1:
                continue
            else:
                sdp.add_constraint(X[i,j] == Cov_ctr[i,j])

    sdp.set_objective('min',0*sum(X))
    sol = sdp.solve(solver = 'cvxopt', verbose = False)
    X = np.array(X.value)

    return X
 
def find_init_sol_b(Cov_ctr, fix):

    shape = Cov_ctr.shape[0]
    sdp = pic.Problem()
    X = sdp.add_variable('X', (shape,shape), vtype='symmetric')
    Cov_ctr = pic.new_param('M', matrix(Cov_ctr))

    #Matrix has to be spd
    sdp.add_constraint(X>>0)

    for i in range(shape-fix, shape):
        for j in range(shape-fix, shape):
          sdp.add_constraint(X[i,j] == Cov_ctr[i,j])

    sdp.set_objective('min', 0*sum(X))
    sol = sdp.solve(solver = 'cvxopt', verbose = True)
    X = np.array(X.value)

    return X

#---------------------------------------------------------------------------
# Compute beta for the naive plug-in estimator
#---------------------------------------------------------------------------

def compute_beta_naive(X,X_l,Y_l,S,alpha,eps,numCauses, X_tr = 0, Y_tr = 0):

    ns_l = X.shape[0]
    ns_s = Y_l.size

    
    numEffects = X.shape[1]-numCauses
    numPredictors = X.shape[1]
    
    cov_x = 1./ns_l*np.dot(X.T,X)
    cov_xs = cov_x[0:numCauses,0:numCauses]
    
    cov_ys = np.dot(cov_xs,alpha)

    cov_yn = 1./ns_s*np.dot(X_l[:,numCauses:].T,Y_l)
    cov_xy = np.append(cov_ys, cov_yn)[:,np.newaxis]
    
    cy = np.dot(alpha[np.newaxis,:],
                np.dot(cov_xs,alpha[:,np.newaxis])) + eps**2
        
    cov_y = np.concatenate([cov_xy,cy]).T
        
    temp = np.append(cov_x,cov_xy.T,axis=0)
    cov = np.append(temp,cov_y.T,axis=1)

    cov_x = cov[0:-1,0:-1]
    cov_xy = cov[-1,0:-1][:,np.newaxis]

    beta_est =  np.dot(np.linalg.inv(cov_x),cov_xy)

    return beta_est

#--------------------------------------------------------------------
# Maximize (4) and return beta
#-------------------------------------------------------------------

def compute_beta_mtl(X,X_l,Y_l,S,alpha,eps,numCauses,X_tr = 0,
                     Y_tr = 0, opti_alpha = False,
                     true_cov = None):

    ns_l = X.shape[0]
    ns_s = Y_l.size
    
    numEffects = X.shape[1]-numCauses
    numPredictors = X.shape[1]

    if true_cov ==None:
        cov_x = np.cov(X.T)
    else:
        cov_x = true_cov[0:-1,0:-1]
        

    if numCauses == numPredictors:
        cov_xs = cov_x[0:numCauses,0:numCauses]
        cov_ys = np.dot(cov_xs,alpha)
        cov_xy = cov_ys[:,np.newaxis]
        cy = np.dot(alpha[np.newaxis,:],
                np.dot(cov_xs,alpha[:,np.newaxis])) + eps**2

        cov_y = np.concatenate([cov_xy,cy]).T
        temp = np.append(cov_x,cov_xy.T,axis=0)
        cov = np.append(temp,cov_y.T,axis=1)

        cov_x = cov[0:-1,0:-1]
        cov_xy = cov[-1,0:-1][:,np.newaxis]
        
        beta_est =  np.dot(np.linalg.inv(cov_x),cov_xy)
        return beta_est

    
    elif numCauses ==0:
        cov_xy = 1./ns_s*np.dot(X_l[:,numCauses:].T,Y_l)
        cov_yn = cov_xy
        cy = np.array([eps**2])[:,np.newaxis]
        
    else:
        cov_xs = cov_x[0:numCauses,0:numCauses]
        cov_ys = np.dot(cov_xs,alpha)
        cov_yn = 1./ns_s*np.dot(X_l[:,numCauses:].T,Y_l)

        cov_xy = np.append(cov_ys, cov_yn)[:,np.newaxis]
        cy = np.dot(alpha[np.newaxis,:],
                    np.dot(cov_xs,alpha[:,np.newaxis])) + eps**2

    
    cov_y = np.concatenate([cov_xy,cy]).T
    temp = np.append(cov_x,cov_xy.T,axis=0)
    x = cov_yn

    M = np.append(temp,cov_y.T,axis=1)
    
    def logl_chol(u):
        
        Mat = M

        Mat[-1,numCauses:-1] = u
        Mat[numCauses:-1,-1] = u.T
        
        try:
            M_inv = np_aut.linalg.inv(Mat)
            det = np_aut.linalg.det(M_inv)
            if np_aut.isnan(det) or det<0:
                log_det = -1e5
            else: log_det = np_aut.log(det)
            
            ret = np_aut.trace(np_aut.dot(M_inv,S)) - log_det
            
        except Exception:
            ret = 1e5
    
        return ret
    
        
    cov= find_init_sol(M,numCauses,numEffects)
    M[-1,numCauses:-1] = cov[-1,numCauses:-1]
    
    x_init = M[-1,numCauses:-1]
    tol= 1e-10
    res = sc.optimize.fmin(logl_chol,x_init,
                            xtol = tol,
                            ftol = tol,
                            maxiter = 1e5,
                            maxfun = 3e5,
                            disp = False)

    M[-1,numCauses:-1] = res
    M[numCauses:-1,-1] = res.T
    
    cov[-1,numCauses:-1] = M[-1,numCauses:-1]
    cov[numCauses:-1,-1] = M[numCauses:-1,-1]    
    cov_x = cov[0:-1,0:-1]
    cov_xy = cov[-1,0:-1][:,np.newaxis]

    beta_est =  np.dot(np.linalg.inv(cov_x),cov_xy)

    return beta_est


#----------------------------------------------------------------------------
#Return MTL coefficient for both the naive and the approach maximizing (4)
#----------------------------------------------------------------------------

def error_naive_beta(train_x, train_y,
                   X_lab,Y_lab,
                   X_ul,test_x,test_y,
                   subset,cov,n,p, alpha=np.zeros(1), eps=0,min_el = 0):

    if eps==0:
        train_x_all = np.append(train_x, X_lab, axis=0)
        train_y_all = np.append(train_y, Y_lab, axis=0)
        regr = linear_model.LinearRegression()
        regr.fit(train_x_all[:,subset],train_y_all)
        pred = regr.predict(train_x_all[:,subset])
        alpha = regr.coef_
        eps = np.std(train_y_all-pred)

    else:
        alpha = alpha
        eps = eps
    
    mask = np.ones(p, dtype = bool)

    mask[subset] = False

    s_size = subset.size

    X_lab_perm = np.concatenate([X_lab[:,subset],X_lab[:,mask]],axis=1)
    X_ul_perm = np.concatenate([X_ul[:,subset],X_ul[:,mask]],axis=1)

    cov = np.concatenate([X_lab_perm, Y_lab], axis=1)
    cov = 1./n*np.dot(cov.T,cov)
                             
    beta = compute_beta_naive(np.append(X_lab_perm, X_ul_perm,axis=0),
                            X_lab_perm,
                            Y_lab,
                            cov,
                            alpha.flatten(),
                            eps,
                            s_size,
                            train_x,
                            train_y)

    bs = beta[0:s_size].flatten()
    bn  = beta[s_size:].flatten()
    
    if min_el != 0:
        n = min_el

    pred_test = np.sum(bs*test_x[:,subset],axis=1) + np.sum(bn*test_x[:,mask],1)
    pred_test = pred_test[:,np.newaxis]
    
    mse_test =np.mean((pred_test-test_y)**2)
    if subset.size > 0 and subset.size<p:
        b = np.zeros(p)
        b[subset] = bs
        b[mask] = bn
    else:
        b = beta.flatten()

    return mse_test, b[:,np.newaxis]


def error_mle_beta(train_x, train_y,
                   X_lab,Y_lab,
                   X_ul,test_x,test_y,
                   subset,cov,n,p, alpha=np.zeros(1), eps=0,min_el = 0,
                   opti_alpha = False,
                   true_cov = None):

    if eps==0:
        train_x_all = np.append(train_x, X_lab, axis=0)
        train_y_all = np.append(train_y, Y_lab, axis=0)
        
        if subset.size > 0:
            regr = linear_model.LinearRegression()
            regr.fit(train_x_all[:,subset],train_y_all)
            pred = regr.predict(train_x_all[:,subset])
            alpha = regr.coef_.flatten()
            eps = np.std(train_y_all-pred)

        else:
            alpha = np.zeros(1)
            eps = np.std(train_y_all)

    else:
        alpha = alpha
        eps = eps

    mask = np.ones(p, dtype = bool)
    if subset.size>0:
        mask[subset] = False

    s_size = subset.size
    if s_size > 0:
        X_lab_perm = np.concatenate([X_lab[:,subset],X_lab[:,mask]],axis=1)
        X_ul_perm = np.concatenate([X_ul[:,subset],X_ul[:,mask]],axis=1)
    else:
        X_lab_perm = X_lab
        X_ul_perm = X_ul

    cov = np.concatenate([X_lab_perm, Y_lab], axis=1)
    cov = np.cov(cov.T)
    
    beta = compute_beta_mtl(np.append(X_lab_perm, X_ul_perm,axis=0),
                            X_lab_perm,
                            Y_lab,
                            cov,
                            alpha.flatten(),
                            eps,
                            s_size,
                            train_x,
                            train_y,
                            opti_alpha,
                            true_cov)

    bs = beta[0:s_size].flatten()
    bn  = beta[s_size:].flatten()
    
    #if min_el != 0:
        #n = min_el

    if subset.size>0:
        pred_test = np.sum(bs*test_x[:,subset],axis=1)+ np.sum(bn*test_x[:,mask],1)
    else: pred_test = np.sum(bn*test_x[:,mask],1)
    pred_test = pred_test[:,np.newaxis]
    
    mse_test =np.mean((pred_test-test_y)**2)

    if subset.size > 0 and subset.size<p:
        b = np.zeros(p)
        b[subset] = bs
        b[mask] = bn
    else:
        b = beta.flatten()

    return mse_test, b[:,np.newaxis]

    
def error_mle_beta_cv(train_x, train_y,
                   X_lab_all, Y_lab_all,
                   X_ul,
                   subset_list,cov,n,p, 
                   alpha=np.zeros(1), eps=0,min_el = 0,
                   opti_alpha = False,
                   true_cov = None):

    scores = []
    fold = 5

    kf = KFold(X_lab_all.shape[0], n_folds = fold)

    for subset in subset_list:
      scores_temp = []

      for train, test in kf:
        X_lab = X_lab_all[train]
        Y_lab = Y_lab_all[train]
        n = X_lab.shape[0]

        if eps==0:
            train_x_all = np.append(train_x, X_lab, axis=0)
            train_y_all = np.append(train_y, Y_lab, axis=0)
            
            if subset.size > 0:
                regr = linear_model.LinearRegression()
                regr.fit(train_x_all[:,subset],train_y_all)
                pred = regr.predict(train_x_all[:,subset])
                alpha = regr.coef_.flatten()
                eps = np.std(train_y_all-pred)

            else:
                alpha = np.zeros(1)
                eps = np.std(train_y_all)

        else:
            alpha = alpha
            eps = eps

        mask = np.ones(p, dtype = bool)
        if subset.size>0:
            mask[subset] = False

    
        s_size = subset.size
        if s_size > 0:
            X_lab_perm = np.concatenate([X_lab[:,subset],X_lab[:,mask]],axis=1)
            X_ul_perm = np.concatenate([X_ul[:,subset],X_ul[:,mask]],axis=1)
        else:
            X_lab_perm = X_lab
            X_ul_perm = X_ul

        cov = np.concatenate([X_lab_perm, Y_lab], axis=1)
        cov = np.cov(cov.T)
        
        beta = compute_beta_mtl(np.append(X_lab_perm, X_ul_perm,axis=0),
                                X_lab_perm,
                                Y_lab,
                                cov,
                                alpha.flatten(),
                                eps,
                                s_size,
                                train_x,
                                train_y,
                                opti_alpha,
                                true_cov)

        bs = beta[0:s_size].flatten()
        bn  = beta[s_size:].flatten()
        
        test_x, test_y = X_lab_all[test], Y_lab_all[test]

        if subset.size>0:
            pred_test = np.sum(bs*test_x[:,subset],axis=1)+ np.sum(bn*test_x[:,mask],1)
        else: pred_test = np.sum(bn*test_x[:,mask],1)
        pred_test = pred_test[:,np.newaxis]
        
        mse_test =np.mean((pred_test-test_y)**2)

        if subset.size > 0 and subset.size<p:
            b = np.zeros(p)
            b[subset] = bs
            b[mask] = bn
        else:
            b = beta.flatten()
        scores_temp.append(mse_test)
        eps = 0
      scores.append(np.mean(scores_temp))

    return subset_list[np.argmin(scores)]


def np_getDistances(x,y):
	K = (x[:,:, np.newaxis] - y.T)
	return np.linalg.norm(K,axis = 1)

def np_gaussian_kernel(x,y, beta=0.1):
    K = np_outer_substract(x,y)
    return np.exp( -beta * np.linalg.norm(K, axis=1))

def mat_hsic(X,nEx):

	nExCum = np.cumsum(nEx)
	domains = np.zeros((np.sum(nEx),np.sum(nEx)))
	currentIndex = 0

	for i in range(nEx.size):

		domains[currentIndex:nExCum[i], currentIndex:nExCum[i]] = np.ones((nEx[i], nEx[i]))
		currentIndex = nExCum[i]

	return domains

def numpy_GetKernelMat(X,sX):

	Kernel = (X[:,:, np.newaxis] - X.T).T
	Kernel = np.exp( -1./(2*sX) * np.linalg.norm(Kernel, axis=1))

	return Kernel

def numpy_HsicGammaTest(X,Y, sigmaX, sigmaY, DomKer = 0):

	n = X.T.shape[1]

	KernelX = numpy_GetKernelMat(X,sigmaX)

	KernelY = DomKer

	coef = 1./n
	HSIC = coef**2*np.sum(KernelX*KernelY) + coef**4*np.sum(
                KernelX)*np.sum(KernelY) - 2*coef**3*np.sum(np.sum(KernelX,axis=1)*np.sum(KernelY, axis=1))
	
	#Get sums of Kernels
	KXsum = np.sum(KernelX)
	KYsum = np.sum(KernelY)

	#Get stats for gamma approx

	xMu = 1./(n*(n-1))*(KXsum - n)
	yMu = 1./(n*(n-1))*(KYsum - n)
	V1 = coef**2*np.sum(KernelX*KernelX) + coef**4*KXsum**2 - 2*coef**3*np.sum(np.sum(KernelX,axis=1)**2)
	V2 = coef**2*np.sum(KernelY*KernelY) + coef**4*KYsum**2 - 2*coef**3*np.sum(np.sum(KernelY,axis=1)**2)

	meanH0 = (1. + xMu*yMu - xMu - yMu)/n
	varH0 = 2.*(n-4)*(n-5)/(n*(n-1.)*(n-2.)*(n-3.))*V1*V2

	#Parameters of the Gamma
	a = meanH0**2/varH0
	b = n * varH0/meanH0

	return n*HSIC, a, b


def sigmoid(x):
	return 1./(1+np.exp(-x))

#--------------------------------------------------------
#Process residuals for computing a Levene test
#-------------------------------------------------------

def levene_pval(Residual,nEx, numR):
	
	prev = 0 
	n_ex_cum = np.cumsum(nEx)

	for j in range(numR):

		r1 = Residual[prev:n_ex_cum[j],:]

		if j == 0:
			residTup = (r1,)

		else:
			residTup = residTup + (r1,)

		prev = n_ex_cum[j]

	return residTup


#----------------------------------------------------------------------------
# Utils for Dica
#--------------------------------------------------------------------------

def get_kernel_mat(x,y, sx2):
    K = (x[:,:, np.newaxis] - y.T)
    return np.exp(-1./(2*sx2)*np.linalg.norm(K,axis = 1)**2)

def get_kernel_mat_lin(x,y, sx2):
    K = (x[:,:, np.newaxis]*y.T)
    return np.sum(K,axis=1)

def np_getDistances(x,y):
	K = (x[:,:, np.newaxis] - y.T)
	return np.linalg.norm(K,axis = 1)


def get_color_dict():

  colors = {
    'pool' : 'red',
    'lasso' : 'red',
    'shat' : 'green',
    'sgreed' : 'green',
    'ssharp' : 'green',
    'strue' : 'blue',
    'cauid' : 'blue',
    'causharp': 'blue',
    'cauul' : 'blue',
    'mean' : 'black',
    'msda' : 'orange',
    'mtl' : 'orange',
    'dica' : 'orange',
    'dom' : 'k',
    'naive' : 'magenta'
  }

  markers = {
    'pool' : 'o',
    'lasso' : '^',
    'shat' : 'o',
    'sgreed' : '^',
    'strue' : '^',
    'ssharp' : 'd',
    'cauid' : 'd',
    'causharp' : 'h',
    'cauul' : '^',
    'mean' : 'o',
    'msda' : 'o',
    'mtl' : '^',
    'dica' : 'd',
    'dom' : 'o',
    'naive' : 'o'
  }

  legends = {
              'pool' : r'$\beta^{CS}$',
              'lasso' : r'$\beta^{CS(\hat S Lasso)}$',
              'shat' : r'$\beta^{CS(\hat S)}$',
              'ssharp' : r'$\beta^{CS(\hat S \sharp)}$',
              'strue' : r'$\beta^{CS(cau)}$',
              'cauid' : r'$\beta^{CS(cau+,id)}$',
              'causharp' : r'$\beta^{CS(cau\sharp)}$',
              'cauul' : r'$\beta^{CS(cau\sharp UL)}$',
              'sgreed' :r'$\beta^{CS(\hat{S}_{greedy})}$',
              'mean'   : r'$\beta^{mean}$',
              'msda'   : r'$\beta^{mSDA}$',
              'mtl'   : r'$\beta^{MTL}$',
              'dica'   : r'$\beta^{DICA}$',
              'naive'   : r'$\beta^{naive}$',
              'dom'   : r'$\beta^{dom}$'
            }

  return colors, markers, legends

def mse(model, x, y):
  return np.mean((model.predict(x)-y)**2)


def intervene_on_p(l_p, sz):
  mask = np.zeros((sz, 1), dtype = bool)
  if len(l_p) > 0:
    mask[l_p] = True
  return mask


def merge_results(f1, f2, key, direc):
  with open(os.path.join(direc, f1), 'rb') as f:
    r1 = pickle.load(f)
  with open(os.path.join(direc, f2), 'rb') as f:
    r2 = pickle.load(f)

  r1['results'][key] = r2['results'][key]
  if key not in r1['plotting'][0]:
      r1['plotting'][0].append(key)

  with open(os.path.join(direc, 'merged.pkl'),'wb') as f:
    pickle.dump(r1, f)


















