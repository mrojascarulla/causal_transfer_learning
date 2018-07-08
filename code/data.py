#!/usr/bin/env python
# Copyright (c) 2014 - 2018  Mateo Rojas-Carulla  [mrojascarulla@gmail.com]
# All rights reserved.  See the file COPYING for license terms.

import numpy as np
import scipy as sc

from scipy.stats import wishart
from sklearn import linear_model
import sklearn.gaussian_process as GP
from sklearn.preprocessing import scale
import sys

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as pl

def linear(X,params):
	alphaC = params[0]
	return alphaC*X #+0.5*X**2

def zero(X,params):
	return 0.


def gen_gauss(mu, sigma, n):
  return np.random.multivariate_normal(mu, sigma, n)

def draw_cov(p):
  scale = np.random.normal(0,1,(p,p))
  scale = np.dot(scale.T, scale)
  if p == 1:
    cov = scale
  else:
    cov = wishart.rvs(df=p, scale=scale)

  #Normalize covariance matrix
  for i in xrange(p):
    for j in xrange(p):
      if i == j: continue
      cov[i,j] = cov[i,j]/np.sqrt(cov[i,i]*cov[j,j])

  np.fill_diagonal(cov,1) 
  return cov

def gen_coef(coef_0, lambd, mask = None):
  if not mask is None:
    mask_compl = ((mask+1)%2).astype(bool)
    draw = np.random.normal(0,1,coef_0.shape)
    ret = (1-lambd)*coef_0 + lambd*draw
    ret[mask_compl] = coef_0[mask_compl]
    return ret
  else:
    return (1-lambd)*coef_0 + lambd*(np.random.normal(0,1,coef_0.shape) + np.random.normal(0,1))

def gen_noise(shape):
  return np.random.normal(0,1,shape)

def covs_all(n_task, p_s, p_n, mask = None):

  cov_s, cov_n = [], []
  fix = -1
  ref = None
  if not mask is None:
    fix_mask = np.where(mask == False)[0]
    if len(fix_mask) > 0:
      fix = fix_mask.size

      ref = draw_cov(fix)
  
  for k in xrange(n_task):
    cov_s.append(draw_cov(p_s))
    cov_n_k = draw_cov(p_n)

    if fix > 0:
      cov_n_k[-fix:, -fix:] = ref 
    eig = np.linalg.eig(cov_n_k)

    if not np.all(eig[0]>0):
      pd = False
      max_iter = 100
      it = 0
      while (not pd) and it<max_iter:
        it += 1
        samp = np.random.normal(0,1,(fix, p_n-fix))
        if np.any(np.array(samp.shape) == 1):
          samp = samp.flatten()
        
        if fix == 1:
          cov_n_k[-fix,0:p_n-fix] = samp.flatten()
          cov_n_k[0:p_n-fix, -fix] = samp.flatten()
        elif fix == p_n-1:
          cov_n_k[-fix:,p_n-fix-1] = samp.flatten()
          cov_n_k[p_n-fix-1, -fix:] = samp.flatten()
        else:
          cov_n_k[-fix:,0:p_n-fix] = samp
          cov_n_k[0:p_n-fix, -fix:] = samp.T

        pd = np.all(np.linalg.eig(cov_n_k)[0]>0) 

    cov_n.append(cov_n_k)

  return cov_s, cov_n

def coefs_all(n_task, p_n, p_conf, lambd, beta_0, gamma_0, mask = None):
  beta, gamma = [], []
  for k in xrange(n_task):
    gamma.append(gen_coef(gamma_0, lambd, mask = mask))
    beta.append(gen_coef(beta_0, lambd))
  return gamma, beta

def draw_tasks(n_task, n, params):

  p_nconf = params['p_nconf']
  mu_s = params['mu_s']
  mu_n = params['mu_n']
  cov_s = params['cov_s']
  cov_n = params['cov_n']
  eps = params['eps']
  alpha = params['alpha']
  beta = params['beta']
  gamma = params['gamma']
  g = params['g']
  x, y, n_ex = [], [], []

  for k in xrange(n_task):
    xs_k    = (gen_gauss(mu_s, cov_s[k], n))
    eps_draw = gen_noise((n,1))
    y_k     = np.dot(xs_k, alpha) + eps*eps_draw
    gamma_k = gamma[k]
    noise_k = (g*gen_gauss(mu_n, cov_n[k], n))
    xn_k = np.dot(y_k, gamma_k.T) + noise_k
    beta_k  = beta[k]

    if p_nconf > 0:
      xn_k += np.dot(xs_k[:,p_nconf:], beta_k)

    x.append(np.concatenate([xs_k, xn_k], 1))
    y.append(y_k)
    n_ex.append(n)

  return np.concatenate(x,0), np.concatenate(y,0), n_ex

def draw_all(alpha, n_task, n, p, p_s, p_conf, eps, g, lambd,beta_0, gamma_0, mask = None):
 
  p_n = p - p_s

  mu_s = np.zeros(p_s)
  mu_n = np.zeros(p_n)
  cov_s, cov_n = covs_all(n_task, p_s, p_n, mask = mask)
  gamma, beta = coefs_all(n_task, p_n, p_conf, lambd, beta_0, gamma_0, mask = mask)
  params = {'mu_s':mu_s, 'mu_n' : mu_n, 'cov_s' : cov_s, 'cov_n' : cov_n, 
            'eps':eps, 'g':g,
            'alpha' : alpha, 'beta': beta, 'gamma': gamma, 'p_nconf' : (p_s-p_conf)}

  x, y, n_ex = draw_tasks(n_task, n, params)
  x_test, y_test, n_ex_test = draw_tasks(n_task, n, params)

  return x, y, x_test, y_test, n_ex, n_ex_test, params

class gauss_tl(object):

  """
  Class for synthetic data experiments.
  """

  def __init__(self, n_task, n, p, p_s, p_conf, eps, g, lambd, lambd_test, mask = None):

    if p_s == p:
      p = p+1
      self.is_full = True
    else:
      self.is_full = False

    p_n = p - p_s
    p_nconf = p_s - p_conf
    alpha = gen_coef(np.random.normal(0,1,(p_s,1)),0)

    gamma_0 = np.random.normal(0,1,(p_n,1))
    beta_0 = np.random.normal(0,1,(p_conf,p_n))

    x, y, x_test, y_test, n_ex, n_ex_test, params = draw_all(alpha, n_task, n, p, p_s, p_conf, eps, 
                                                              g, lambd,
                                                              beta_0, gamma_0, mask = mask)

    xt, yt, x_tt, y_tt, n_ext, n_ex_tt, params_test = draw_all(alpha, n_task, 
                                                                n, p, p_s, p_conf, eps, g, lambd_test, 
                                                                beta_0, gamma_0, mask = mask)

    if self.is_full:
      x = x[:,0:-1]
      x_test = x_test[:,0:-1]
      xt = xt[:,0:-1]
      x_tt = x_tt[:,0:-1]
      self.p = p-1
      self.alpha = alpha[:-1]
    else:
      self.p = p
      self.alpha = alpha

    self.p_s = p_s
    self.p_conf = p_conf
    self.train = {}
    self.train['x_train'] = x
    self.train['y_train'] = y
    self.train['x_test'] = x_test
    self.train['y_test'] = y_test
    self.train['n_ex'] = np.array(n_ex)
    self.train['n_ex_test'] = np.array(n_ex_test)
    self.train['cov_s'] = params['cov_s']
    self.train['cov_n'] = params['cov_n']
    self.train['eps'] = params['eps']
    self.train['gamma'] = params['gamma']
    self.train['beta'] = params['beta']
    self.lambd = lambd
    self.lambd_test = lambd_test
    self.g = g
    self.eps = eps

    self.test = {}
    self.test['x_train'] = xt
    self.test['y_train'] = yt
    self.test['x_test'] = x_tt
    self.test['y_test'] = y_tt
    self.test['n_ex'] = np.array(n_ext)
    self.test['n_ex_test'] = y_tt
    self.test['cov_s'] = params_test['cov_s']
    self.test['cov_n'] = params_test['cov_n']
    self.test['eps'] = params_test['eps']
    self.test['gamma'] = params_test['gamma']
    self.test['beta'] = params_test['beta']
    self.gamma_0 = gamma_0
    self.beta_0 = beta_0
    self.n_ex = np.array(n_ext)
    self.n_ex_test = np.array(n_ex_tt)
    self.n_task = n_task
    self.n = n

  def resample(self,n_task, n, g = None, lambd = None, eps = None, noise = 0, mask = None):

    if g is None: g = self.g
    if eps is None: eps = self.eps
    if lambd is None: lambd = self.lambd

    alpha = gen_coef(np.random.normal(0,1,(self.p_s,1)),0)

    xt, yt, x_tt, y_tt, n_ext, n_ex_tt, params = draw_all(alpha, 
                                                  self.n_task, self.n, 
                                                  self.p, self.p_s, self.p_conf, 
                                                  eps, g,lambd, 
                                                  self.beta_0, self.gamma_0, 
                                                  mask = mask)

    
    xt_test, yt_test, x_tt_test, y_tt_test, n_ext, n_ex_tt, params_test = draw_all(alpha, 
                                                  self.n_task, self.n, 
                                                  self.p, self.p_s, self.p_conf, 
                                                  eps, g,self.lambd_test, 
                                                  self.beta_0, self.gamma_0, 
                                                  mask = mask)
    if noise> 0:
      xt = np.append(xt, np.random.normal(0,1,(xt.shape[0],noise)),1)

    self.alpha = alpha
    self.train['x_train'] = xt
    self.train['y_train'] = yt
    self.train['x_test'] = x_tt
    self.train['y_test'] = y_tt
    self.train['n_ex'] = np.array(n_ext)
    self.train['n_ex_test'] = np.array(n_ex_tt)
    self.train['cov_s'] = params['cov_s']
    self.train['cov_n'] = params['cov_n']
    self.train['eps'] = params['eps']
    self.train['gamma'] = params['gamma']
    self.train['beta'] = params['beta']

    self.test['x_test'] = xt_test
    self.test['y_test'] = yt_test
    self.test['cov_s'] = params_test['cov_s']
    self.test['cov_n'] = params_test['cov_n']
    self.test['eps'] = params_test['eps']
    self.test['gamma'] = params_test['gamma']
    self.test['beta'] = params_test['beta']
    self.lambd = lambd
    self.g = g
    self.eps = eps

    return xt, yt

  def add_noise(self,n_noise):
    n = self.train['x_train'].shape[0]
    n_t = self.train['x_train'].shape[0]
    n_b = self.train['x_train'].shape[0]
    n_bt = self.train['x_train'].shape[0]
    self.train['x_train'] = np.append(self.train['x_train'], np.random.normal(0,1,(n,n_noise)),1)
    self.train['x_test'] = np.append(self.train['x_test'], np.random.normal(0,1,(n_t,n_noise)),1)
    self.test['x_train'] = np.append(self.test['x_train'], np.random.normal(0,1,(n_b,n_noise)),1)
    self.test['x_test'] = np.append(self.test['x_test'], np.random.normal(0,1,(n_bt,n_noise)),1)


  def true_cov(self, train = True):

    if train:
      cov_s_all = self.train['cov_s'] 
      cov_n_all = self.train['cov_n']
      gamma_all = self.train['gamma']
      beta_all = self.train['beta']
    else:
      cov_s_all = self.test['cov_s'] 
      cov_n_all = self.test['cov_n']
      gamma_all = self.test['gamma']
      beta_all = self.test['beta']
    
    alpha = self.alpha
    p_s = self.p_s
    if self.is_full:
      p_n = self.p+1 - p_s
    else:
      p_n = self.p - p_s
    cov_mats = []

    for t in xrange(len(cov_s_all)):
      if self.is_full:
        cov_n = sc.linalg.block_diag(cov_s_all[t], np.array(self.train['eps']**2))
      else:
        cov_n = sc.linalg.block_diag(cov_s_all[t],self.g**2*cov_n_all[t])
        cov_n = sc.linalg.block_diag(cov_n, np.array(self.train['eps']**2))

      B = np.eye(p_s)

      if not self.is_full:
        if self.p_conf>0:
          beta_fill = np.append(np.zeros((p_n,p_s-self.p_conf)), beta_all[t].T,1)
        else:
          beta_fill = np.zeros((p_n,p_s))

        Bn = np.append(beta_fill, np.zeros((p_n, p_n)), 1)
        Bn = np.append(Bn,gamma_all[t],axis=1)

      By = np.append(alpha.T,np.zeros((1,p_n+1)),axis=1)
      Bs = np.zeros((p_s, self.p + 1))
      if self.is_full:
        B = np.concatenate([Bs, By],axis=0)
      else:
        B = np.concatenate([Bs,Bn,By],axis=0)

      ImB = np.linalg.inv(np.eye(B.shape[0])-B)

      cov_mats.append(np.dot(ImB,np.dot(cov_n,ImB.T)))

    return cov_mats

