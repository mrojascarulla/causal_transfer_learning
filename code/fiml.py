#!/usr/bin/env python
# Copyright (c) 2014 - 2018  Mateo Rojas-Carulla  [mrojascarulla@gmail.com]
# All rights reserved.  See the file COPYING for license terms.

import autograd.numpy as np
from autograd import grad
from sklearn.cross_validation import KFold
from sklearn import linear_model

def fiml(sigma_chol, n_predictors, mask, mask_var,X,Y, sub,n_ul = 0, 
         print_val = False):
    """
    Implements full information maximum likelihood for optimization.
    """
    sigma_chol = sigma_chol.reshape(n_predictors+1,n_predictors+1)
    sigma = np.dot(sigma_chol, sigma_chol.T)

    test_x, test_y = X[mask], Y[mask]

    mask_train = ((mask+1)%2).astype(bool)
    if n_ul>0:
        mask_train[0:n_ul] = False
    train_x, train_y = X[mask_train], Y[mask_train]
    missing = ((mask_var+1)%2).astype(bool)

    joint_test = np.concatenate([test_y, test_x],axis=1)
    samp_cov = np.cov(joint_test.T)

    joint_train = np.concatenate([train_y, train_x], axis=1)
    samp_cov_t = np.cov(joint_train[:,sub].T)

    L = np.linalg.solve(sigma, samp_cov)
    L = -np.trace(L)
    L -= np.log(np.linalg.det(sigma))
    L *= test_x.shape[0]

    det_sub = np.linalg.det(sigma[sub].T[sub].T)

    L_tr = np.linalg.solve(sigma[sub].T[sub].T,samp_cov_t)
    L_tr = -np.trace(L_tr)
    L_tr -= np.log(det_sub)
    L_tr *= train_x.shape[0]

    if n_ul > 0:
        set_n = np.arange(1,n_predictors+1)
        joint_ul = X[0:n_ul,:]#np.concatenate([Y[0:n_ul,:], X[0:n_ul,1:]], axis=1)

        samp_cov_t = np.cov(joint_ul.T)
        mask_ul = np.copy(mask_train)
        mask_ul[0:n_ul] = True
        mask_ul[n_ul:] = False
        det_sub = np.linalg.det(sigma[1:,1:])

        L_ul = -np.trace(np.dot(np.linalg.inv(sigma[1:,1:]),samp_cov_t))
        L_ul -= np.log(det_sub)
        L_ul *= n_ul
    else:
        L_ul = 0

    if print_val:
      print -(L + L_tr-L_ul)

    return -(L+L_tr-L_ul)


def e_step(sigma,X, mask_var, mask_samp,n_ul = 0):

    """
    E step for MTL algorithm.
    """
    n = X.shape[0]
    n_tr = np.size(np.where(mask_samp == False)[0])-n_ul

    mask_var_c = ((mask_var+1)%2).astype(bool)
    mask_samp_c = ((mask_samp+1)%2).astype(bool)
    
    if n_ul>0:
        mask_samp_c[0:n_ul] = False

    sigma_obs = sigma[mask_var].T[mask_var].T
    sigma_cond = sigma[mask_var].T[mask_var_c].T
    mat_prod = sigma_cond.T.dot(np.linalg.inv(sigma_obs))

    mu_upd =mat_prod.dot(X[mask_samp_c].T[mask_var])
    m = mask_samp_c[:,None]*mask_var_c[None,:]

    X[m] = mu_upd.T.flatten()        
   
    #compute sigma
    sigma_miss = sigma[mask_var_c].T[mask_var_c].T
    sigma_upd = sigma_miss - mat_prod.dot(sigma_cond)

    #print sigma_upd

    if n_ul>0:

        mask_vul = np.copy(mask_var)
        mask_vul[0] = False
        mask_vul[1:] = True

        mask_vul_c = ((mask_vul+1)%2).astype(bool)
        sigma_obs_ul = sigma[mask_vul].T[mask_vul].T
        sigma_cond_ul = sigma[mask_vul].T[mask_vul_c].T

        mask_samp_ul = np.zeros(mask_samp.shape,dtype = bool)
        mask_samp_ul[0:n_ul] = True
        
        mat_prod_ul = sigma_cond_ul.T.dot(np.linalg.inv(sigma_obs_ul))
        mu_upd =mat_prod_ul.dot(X[mask_samp_ul].T[mask_vul])

        m = mask_samp_ul[:,None]*mask_vul_c[None,:]
        #X[m] =-0.65768#
        #X[m] = ex.flatten()#
        X[m] = mu_upd.T.flatten()

        #X[0:n_ul,0] = ex.flatten()
        sigma_miss_ul = sigma[mask_vul_c].T[mask_vul_c].T
        sigma_upd_ul = sigma_miss_ul - mat_prod_ul.dot(sigma_cond_ul)
        m_ul = mask_vul_c[:,None]*mask_vul_c[None,:]

    sigma_new = np.cov(X.T)

    #exit()
    m = mask_var_c[:,None]*mask_var_c[None,:]
    sigma_new[m] += sigma_upd.flatten()*n_tr/float(n)

    if n_ul>0:
        sigma_new[m_ul] +=n_ul/float(n)*sigma_upd_ul.flatten()

    return sigma_new

def m_step(sigma):
    """
    Perform M step for MTL. 
    """
    sigma_new = sigma - mu_new[:,None].dot(mu_new[None,:])

    return sigma_new
    
def subset_cv(sub_list, test_x, test_y,
              train_x, train_y,
              samp,
              num_predictors,
              n_ul = 0,
              x_ul = 0):

    """
    In MTL, preturn subset using cross-validation.
    """
    scores = []
    fold = 2

    kf = KFold(test_x.shape[0],n_folds = fold)
    
    for s in sub_list:
        scores_temp = []

        for train, test in kf:
            test_x_cv = test_x[train]
            test_y_cv = test_y[train]

            X = np.concatenate([test_x_cv, train_x],axis=0)
            Y = np.concatenate([test_y_cv, train_y],axis=0)
            app_xy = np.append(Y,X,axis=1)
            

            mask = np.zeros(app_xy.shape[0], dtype = bool)

            mask[0:test_x_cv.shape[0]] = True
            mask_var = np.zeros(app_xy.shape[1],dtype=bool)
            mask_var[0] = True

            if s.size>0:
                mask_var[s+1] = True
            
            app_xyt = np.concatenate([test_y_cv, test_x_cv],axis=1)
            sigma = np.cov(app_xyt.T) +2/(np.log(test_x_cv.shape[0])**2)*np.eye(app_xyt.shape[1]) 

            index=0
            stay = True
            
            while stay:
                sigma = e_step(sigma,app_xy,mask_var, mask)
                stay =  (index<20)
                index += 1
                
            cov_xsh = sigma[1:,1:]
            cov_xysh = sigma[0,1:][:,np.newaxis]

            beta_cs =  np.dot(np.linalg.inv(cov_xsh),cov_xysh)
            scores_temp.append(np.mean((test_x[test].dot(beta_cs)-test_y[test])**2))
        scores.append(np.mean(scores_temp))
    return sub_list[np.argmin(scores)]



