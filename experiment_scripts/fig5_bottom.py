#/usr/bin/env python
# Copyright (c) 2014 - 2016  Mateo Rojas-Carulla  [mr597@cam.ac.uk]
# All rights reserved.  See the file COPYING for license terms.

from os.path import join
import sys
import glob, json

import time
import cPickle as pickle
import argparse

import os
sys.path.append('../code')
import data
import timeit
import subset_search
import utils
import mSDA
import fiml
import plotting

from sklearn import linear_model
from sklearn.cross_validation import KFold

from scipy import optimize as sopt
import scipy.io as io

import autograd.numpy as np
from autograd import grad, jacobian

#import matlab_wrapper

np.random.seed(1234)

#Given a number of training tasks, the total number of examples and the number per task, return a boolean mask
def mask_training_tasks(n_tasks,n_s,n_tot,n_pred):
    mask = np.zeros((n_tot,n_pred),dtype = bool)
    n_each = n_tot/n_tasks
    for t in range(n_tasks):
        mask[t*n_each:t*n_each+n_s,:] = True
    return mask


def fiml_(sigma_chol,mask_samp,mask_var):
    
    sigma_chol = sigma_chol.reshape(n_predictors+1,n_predictors+1)
    sigma = np.dot(sigma_chol, sigma_chol.T)

    test_x, test_y = X[mask_samp], Y[mask_samp]
    mask_train = ((mask+1)%2).astype(bool)
    train_x, train_y = X[mask_train], Y[mask_train]
    missing = ((mask_var+1)%2).astype(bool)

    joint_test = np.concatenate([test_y, test_x],axis=1)
    samp_cov = np.cov(joint_test.T)

    joint_train = np.concatenate([train_y, train_x], axis=1)

    joint_train.T[missing] = 0
    samp_cov_t = np.cov(joint_train[:,0:n_causes+1].T)

    L = -np.trace(np.dot(np.linalg.inv(sigma),samp_cov))
    L -= np.log(np.linalg.det(sigma))
    L *= test_x.shape[0]

    det_sub = np.linalg.det(sigma[0:n_causes+1,0:n_causes+1])

    L_tr = -np.trace(np.dot(np.linalg.inv(sigma[0:n_causes+1,0:n_causes+1]),samp_cov_t))
    L_tr -= np.log(det_sub)
    L_tr *= train_x.shape[0]

    return -(L+L_tr)

def fiml_auto(sigma_chol):
    
    sigma_chol = sigma_chol.reshape(n_predictors+1,n_predictors+1)
    sigma = np.dot(sigma_chol, sigma_chol.T)

    test_x, test_y = X[mask], Y[mask]
    mask_train = ((mask+1)%2).astype(bool)
    train_x, train_y = X[mask_train], Y[mask_train]
    missing = ((mask_var+1)%2).astype(bool)

    joint_test = np.concatenate([test_y, test_x],axis=1)
    samp_cov = np.cov(joint_test.T)

    joint_train = np.concatenate([train_y, train_x], axis=1)
    samp_cov_t = np.cov(joint_train[:,0:n_causes+1].T)

    L = -np.trace(np.dot(np.linalg.inv(sigma),samp_cov))
    L -= np.log(np.linalg.det(sigma))
    L *= test_x.shape[0]

    det_sub = np.linalg.det(sigma[0:n_causes+1,0:n_causes+1])

    L_tr = -np.trace(np.dot(np.linalg.inv(sigma[0:n_causes+1,0:n_causes+1]),samp_cov_t))
    L_tr -= np.log(det_sub)
    L_tr *= train_x.shape[0]

    return -(L+L_tr)

np.random.seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', default = '../results')
parser.add_argument('--n_task', default=7)
parser.add_argument('--merge_mtl', default=0)
parser.add_argument('--n', default=10000)
parser.add_argument('--p', default = 6)
parser.add_argument('--p_s', default = 3)
parser.add_argument('--p_conf', default = 0)
parser.add_argument('--eps', default = 2)
parser.add_argument('--g', default = 1)
parser.add_argument('--lambd', default = 0.5)
parser.add_argument('--lambd_test', default = 0.99)
parser.add_argument('--use_hsic', default = 0)
parser.add_argument('--alpha_test', default = 0.05)
parser.add_argument('--n_repeat', default = 20)
parser.add_argument('--max_l', default = 1000)
parser.add_argument('--min_l', default = 50)
parser.add_argument('--n_ul', default = 100)
args = parser.parse_args()

save_dir = args.save_dir

if not os.path.exists(save_dir):
  os.makedirs(save_dir)

save_dir = os.path.join(save_dir, 'fig5_bottom')
if not os.path.exists(save_dir):
  os.makedirs(save_dir)

#If only plot is true, loads results, merges with MTL and plots again. 
if int(args.merge_mtl) == 1:
    utils.merge_results('results.pkl', 'mtl.pkl', 'mtl', save_dir)
    plotting.plot_mtl(os.path.join(save_dir, 'merged.pkl'))
    plotting.plot_mtl_mse(os.path.join(save_dir, 'merged.pkl'))
    exit()

n_task = int(args.n_task)
n = int(args.n)

p = int(args.p)
p_s = int(args.p_s)
p_conf = int(args.p_conf)
eps = float(args.eps)
g = float(args.g)

lambd = float(args.lambd)
lambd_test = float(args.lambd_test)

alpha_test = float(args.alpha_test)
use_hsic = bool(int(args.use_hsic))

dataset = data.gauss_tl(n_task, n, p, p_s, p_conf, eps , g, lambd, lambd_test)

n_repeat = int(args.n_repeat)
max_l = int(args.max_l)
min_l = int(args.min_l)
n_ul = int(args.n_ul)
step_samp = (max_l + 1 - min_l) / 10
samp_array = np.arange(min_l, max_l, step_samp)
results = {}

run_cv = True

methods = [
          'pool',
          'dom',
          'msda',
          'causharp',
          'ssharp',
          'mtl'
        ]


color_dict, markers, legends = utils.get_color_dict()
for m in methods:
  results[m]  = np.zeros((n_repeat, samp_array.size, n_task))

def save(results):
  save_all = {}
  save_all['results'] = results
  save_all['plotting'] = [methods, color_dict, legends, markers]
  save_all['n_samp'] = samp_array

  with open(os.path.join(save_dir, 'results.pkl'),'wb') as f:
    pickle.dump(save_all, f)


for i in range(n_repeat):
    print i
    x_train, y_train = dataset.resample(n_task, n)
    n_ex = dataset.n_ex
    n_ex_c = np.append(0, np.cumsum(n_ex))

    train_x_lab, train_y_lab = [], []
    train_x_ul = []
    x_test, y_test = dataset.train['x_test'], dataset.train['y_test']

    for t in range(n_task):
        train_x_lab.append(x_train[n_ex_c[t]:n_ex_c[t]+max_l,:])
        train_y_lab.append(y_train[n_ex_c[t]:n_ex_c[t]+max_l,:])
        train_x_ul.append(x_train[n_ex_c[t]+max_l:n_ex_c[t]+max_l+n_ul,:])

    train_x_lab_a = np.concatenate(train_x_lab, axis=0)
    train_y_lab_a = np.concatenate(train_y_lab, axis=0)
    train_x_ul_a = np.concatenate(train_x_ul, axis=0)

    for s in range(samp_array.size):

        samp = samp_array[s]
        draw_train = samp
        mask = mask_training_tasks(n_task, draw_train, train_x_lab_a.shape[0], 
                                   p)

        train_x = train_x_lab_a[mask].reshape((n_task * draw_train, p))
        train_y = train_y_lab_a[mask[:, 0]].reshape((n_task * draw_train, 1))

        sub_list = subset_search.subset(train_x, train_y, 
                                        np.array(n_task * [samp]), 
                                        delta=alpha_test,
                                        valid_split=0.6,
                                        use_hsic=False,
                                        return_n_best=3)

        if len(sub_list) == 0:
          sub_list.append(np.array([]))

        #--------------------------------------------
        #Errors with DA
        #--------------------------------------------

        train_cov_mats = dataset.true_cov()
        ns = train_x.shape[0]
        probas = np.linspace(0.01, 1, 10)
        times = []

        p_cv = mSDA.mSDA_cv(probas, train_x, train_y, n_cv = 10)

        W,train_f =mSDA.mSDA(train_x.T, p_cv, 1)
        
        doms, subs = [], []

        for t in range(n_task):
            train_x_task = train_x_lab[t][0:samp, :]
            train_y_task = train_y_lab[t][0:samp, :]
            test_x_task = x_test[t * n:(t + 1) * n]
            test_y_task = y_test[t * n:(t + 1) * n]

            reg = linear_model.LinearRegression()
            reg.fit(train_x_task, train_y_task)

            cov_test = train_cov_mats[t]
            cov_x = cov_test[0:-1, 0:-1]
            cov_xy = cov_test[-1, 0:-1][:, np.newaxis]

            beta_best =  np.dot(np.linalg.inv(cov_x), cov_xy)

            regr = linear_model.LinearRegression()
            regr.fit(train_f[-1].T[t*samp:(t+1)*samp,:], 
                     train_y[t*samp:(t+1)*samp,:])
            diff_beta = np.append(regr.coef_,
                                  np.zeros(p-regr.coef_.size)).reshape(beta_best.shape)-beta_best

            mse_mda = (diff_beta.T).dot(cov_x.dot(diff_beta))
            results['msda'][i, s, t] += mse_mda

            #------------------------------------------------------------------
            # beta dom
            #------------------------------------------------------------------
            regr_dom = linear_model.LinearRegression()
            regr_dom.fit(train_x_task, train_y_task)
            diff_beta = (regr_dom.coef_.reshape(beta_best.shape) - beta_best)

            dom_temp = (diff_beta.T).dot(cov_x.dot(diff_beta))
            doms.append(dom_temp)
            results['dom'][i, s, t] = dom_temp

            #------------------------------------------------------------------
            # beta pool
            #------------------------------------------------------------------
            regr_pool = linear_model.LinearRegression()
            regr_pool.fit(train_x, train_y)

            reg.fit(train_x, train_y)

            diff_beta = (regr_pool.coef_.reshape(beta_best.shape) - beta_best)
            pool_temp = (diff_beta.T).dot(cov_x.dot(diff_beta))

            results['pool'][i, s, t] = pool_temp

            true_causal_set = np.arange(p_s)

            #------------------------------------------------------------------
            # beta  cau sharp
            #------------------------------------------------------------------

            X, Y = np.copy(train_x), np.copy(train_y)
            app_xy = np.append(Y,X,axis=1)

            mask = np.zeros(train_x.shape[0], dtype = bool)
            mask[t * samp:(t + 1) * samp] = True
            mask_var = np.zeros(train_x.shape[1] + 1, dtype=bool)
            mask_var[0:p_s + 1] = True

            app_xyt = np.concatenate([train_y_task, train_x_task],axis=1)
            mu = np.mean(app_xyt, axis=0)
            sigma = np.cov(app_xyt.T) + 2 / (np.log(samp)**2)*np.eye(app_xyt.shape[1])
            old_val = 1e5
            prev = 1e10

            index=0
            stay = True
            while stay:
                sigma = fiml.e_step(sigma, app_xy, mask_var, mask)

                stay =  (index<100)
                index+=1

            cov_xsh = sigma[1:, 1:]
            cov_xysh = sigma[0, 1:][:, np.newaxis]

            beta_cs =  np.dot(np.linalg.inv(cov_xsh), cov_xysh)
            diff_beta = (beta_cs.reshape(beta_best.shape) - beta_best)
            sharp_res = (diff_beta.T).dot(cov_x.dot(diff_beta))

            results['causharp'][i, s, t] = sharp_res

            #------------------------------------------------------------------
            # beta  S sharp
            #------------------------------------------------------------------
            train_x_cv = np.concatenate([train_x[0:samp * t], 
                                         train_x[samp * (t + 1):]], 0)
            train_y_cv = np.concatenate([train_y[0:samp * t], 
                                         train_y[samp * (t + 1):]], 0)
            sub_full = fiml.subset_cv(sub_list,
                                      train_x_task, train_y_task,
                                      train_x_cv, train_y_cv,
                                      samp,
                                      p)

            X, Y = np.copy(train_x), np.copy(train_y)
            app_xy = np.append(Y, X, axis=1)

            mask = np.zeros(train_x.shape[0], dtype = bool)
            mask[t * samp:(t + 1) * samp] = True
            mask_var = np.zeros(train_x.shape[1] + 1, dtype=bool)
            if sub_full.size>0:  mask_var[(sub_full + 1)] = True

            mask_var[0] = True
            
            app_xyt = np.concatenate([train_y_task, train_x_task], axis=1)
            mu = np.mean(app_xyt, axis=0)
            sigma = np.cov(app_xyt.T) + 2/(np.log(samp) ** 
                      2) * np.eye(app_xyt.shape[1])

            index=0
            stay = True
            if sub_full.size>0: sub_full = np.concatenate([[0], sub_full + 1]).astype(int)
            else: sub_full = np.array([0])
            
            while stay:
                sigma = fiml.e_step(sigma,app_xy,mask_var, mask)
                stay =  (index<100)
                index+=1

            cov_xsh = sigma[1:,1:]
            cov_xysh = sigma[0,1:][:,np.newaxis]
            beta_cs =  np.dot(np.linalg.inv(cov_xsh),cov_xysh)

            diff_beta = (beta_cs.reshape(beta_best.shape)-beta_best)
            hatsharp_res = (diff_beta.T).dot(cov_x.dot(diff_beta))

            results['ssharp'][i, s, t] = hatsharp_res

#Save results and plot
save(results)
plotting.plot_mtl(os.path.join(save_dir, 'results.pkl'))
plotting.plot_mtl_mse(os.path.join(save_dir, 'results.pkl'))

