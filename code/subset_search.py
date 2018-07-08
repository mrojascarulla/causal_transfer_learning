#!/usr/bin/env python
# Copyright (c) 2014 - 2018  Mateo Rojas-Carulla  [mrojascarulla@gmail.com]
# All rights reserved.  See the file COPYING for license terms.

import numpy as np
import scipy as sp
from sklearn import linear_model
from sklearn import svm
from sklearn import metrics
import itertools
import utils
import sys

def full_search(train_x, train_y, valid_x, valid_y, n_ex, n_ex_valid, 
                use_hsic, alpha, return_n_best=None):
    """
    Perform Algorithm 1, search over all possible subsets of features. 

    Args:
        dataset: internal dataset object.
        use_hsic: whether to use HSIC. If not, Levene test is used.
        alpha: level for the statistical test of equality of distributions 
          (HSIC or Levene).
        return_n_best: return top n subsets (in terms of test statistic). 
          Default returns only the best subset. 

    """
    num_tasks = len(n_ex)
    n_ex_cum = np.cumsum(n_ex)

    index_task = 0
    best_subset = []
    accepted_sets = []
    accepted_mse = []
    all_sets = []
    all_pvals = []

    num_s = np.sum(n_ex)
    num_s_valid = np.sum(n_ex_valid)
    best_mse = 1e10

    rang = np.arange(train_x.shape[1])
    maxevT = -10
    maxpval = 0
    num_accepted = 0
    current_inter = np.arange(train_x.shape[1])

    #Get numbers for the mean
    pred_valid = np.mean(train_y)
    residual = valid_y - pred_valid

    if use_hsic:
        valid_dom = utils.mat_hsic(valid_y, n_ex_valid)
        ls = utils.np_getDistances(residual, residual)
        sx = 0.5 * np.median(ls.flatten())
        stat, a, b = utils.numpy_HsicGammaTest(residual, valid_dom,
                                               sx, 1, 
                                               DomKer = valid_dom)
        pvals = 1. - sp.stats.gamma.cdf(stat, a, scale=b)
    else:
        residTup = utils.levene_pval(residual, n_ex, num_tasks)
        pvals = sp.stats.levene(*residTup)[1]

    if (pvals > alpha):
        mse_current  = np.mean((valid_y - pred_valid) ** 2)
        if mse_current < best_mse:
            best_mse = mse_current
            best_subset = []
            accepted_sets.append([])
            accepted_mse.append(mse_current)
    
    all_sets.append([])
    all_pvals.append(pvals)

    for i in range(1, rang.size + 1):
        for s in itertools.combinations(rang, i):
            currentIndex = rang[np.array(s)]
            regr = linear_model.LinearRegression()
            
            #Train regression with given subset on training data
            regr.fit(train_x[:, currentIndex], 
                     train_y.flatten())

            #Compute mse for the validation set
            pred = regr.predict(
              valid_x[:, currentIndex])[:,np.newaxis]

            #Compute residual
            residual = valid_y - pred

            if use_hsic:
                valid_dom = utils.mat_hsic(valid_y, n_ex_valid)
                ls = utils.np_getDistances(residual, residual)
                sx= 0.5 * np.median(ls.flatten())
                stat, a, b = utils.numpy_HsicGammaTest(
                    residual, valid_dom, sx, 1, DomKer = valid_dom)
                pvals = 1.- sp.stats.gamma.cdf(stat, a, scale=b)
            else:
                residTup = utils.levene_pval(residual, n_ex_valid, num_tasks)
                pvals = sp.stats.levene(*residTup)[1]
            
            all_sets.append(s)
            all_pvals.append(pvals)
                                                                            
            if (pvals > alpha):
                mse_current = np.mean((pred - valid_y) ** 2)
                if mse_current < best_mse: 
                    best_mse = mse_current
                    best_subset = s
                    current_inter = np.intersect1d(current_inter, s)
                    accepted_sets.append(s)
                    accepted_mse.append(mse_current)


    if len(accepted_sets) == 0:
        all_pvals = np.array(all_pvals).flatten()
        sort_pvals = np.argsort(all_pvals)
        idx_max = sort_pvals[-1]
        best_subset = all_sets[idx_max]
        accepted_sets.append(best_subset)

    if return_n_best:
        return [np.array(s) for s in accepted_sets[-return_n_best:]]
    else:
        return np.array(best_subset)

def greedy_search(train_x, train_y, valid_x, valid_y, n_ex, n_ex_valid, 
                  use_hsic, alpha, inc = 0.0):

    num_s = np.sum(n_ex)

    num_predictors = train_x.shape[1]
    best_subset = np.array([])
    best_subset_acc = np.array([])
    best_mse_overall = 1e10

    already_acc = False

    selected = np.zeros(num_predictors)
    accepted_subset = None

    all_sets, all_pvals = [], []

    n_iters = 10*num_predictors
    stay = 1

    pow_2 = np.array([2**i for i in np.arange(num_predictors)])

    ind = 0
    prev_stat = 0

    bins = []
   
    #Get numbers for the mean

    pred = np.mean(train_y)
    mse_current = np.mean((pred - valid_y) ** 2)
    residual = valid_y - pred

    residTup = utils.levene_pval(residual, n_ex_valid, n_ex_valid.size)
    levene = sp.stats.levene(*residTup)

    all_sets.append(np.array([]))
    all_pvals.append(levene[1])
    if all_pvals[-1]>alpha:
      accepted_subset = np.array([])

    while (stay==1):
        
        pvals_a = np.zeros(num_predictors)
        statistic_a = 1e10 * np.ones(num_predictors)
        mse_a = np.zeros(num_predictors)
    
        for p in range(num_predictors):
            current_subset = np.sort(np.where(selected == 1)[0])
            regr = linear_model.LinearRegression()
            
            if selected[p]==0:
                subset_add = np.append(current_subset, p).astype(int)
                regr.fit(train_x[:,subset_add], train_y.flatten())
                
                pred = regr.predict(valid_x[:,subset_add])[:,np.newaxis]
                mse_current = np.mean((pred - valid_y)**2)
                residual = valid_y - pred

                residTup = utils.levene_pval(residual,n_ex_valid,
                                                     n_ex_valid.size)
                
                levene = sp.stats.levene(*residTup)

                pvals_a[p] = levene[1]
                statistic_a[p] = levene[0]
                mse_a[p] = mse_current

                all_sets.append(subset_add)
                all_pvals.append(levene[1])
                
            if selected[p] == 1:
                acc_rem = np.copy(selected)
                acc_rem[p] = 0

                subset_rem = np.sort(np.where(acc_rem == 1)[0])

                if subset_rem.size ==0: continue
                
                regr = linear_model.LinearRegression()
                regr.fit(train_x[:,subset_rem], train_y.flatten())

                pred = regr.predict(valid_x[:,subset_rem])[:,np.newaxis]
                mse_current = np.mean((pred - valid_y)**2)
                residual = valid_y - pred
                
                residTup = utils.levene_pval(residual,n_ex_valid, 
                                                     n_ex_valid.size)
                levene = sp.stats.levene(*residTup)
                
                pvals_a[p] = levene[1]
                statistic_a[p] = levene[0]
                mse_a[p] = mse_current

                all_sets.append(subset_rem)
                all_pvals.append(levene[1])

        accepted = np.where(pvals_a > alpha)

        if accepted[0].size>0:
            best_mse = np.amin(mse_a[np.where(pvals_a > alpha)])
            already_acc = True

            selected[np.where(mse_a == best_mse)] = \
              (selected[np.where(mse_a == best_mse)] + 1) % 2

            accepted_subset = np.sort(np.where(selected == 1)[0])
            binary = np.sum(pow_2 * selected)
   
            if (bins==binary).any():
                stay = 0
            bins.append(binary)
        else:
            best_pval_arg = np.argmin(statistic_a)

            selected[best_pval_arg] = (selected[best_pval_arg] + 1) % 2
            binary = np.sum(pow_2 * selected)

            if (bins==binary).any():
                stay = 0
            bins.append(binary)

        if ind>n_iters:
            stay = 0
        ind += 1

    if accepted_subset is None:
      all_pvals = np.array(all_pvals).flatten()

      max_pvals = np.argsort(all_pvals)[-1]
      accepted_subset = np.sort(all_sets[max_pvals])

    return np.array(accepted_subset)

def subset(x, y, n_ex, delta, valid_split, use_hsic = False, 
           return_n_best = None):

    """
    Run Algorithm 1 for full subset search. 

    Args:
        x: train features. Shape [n_examples, n_features].
        y: train labels. Shape [n_examples, 1].
        n_ex: list with number of examples per task (should be ordered in 
          train_x and train_y). Shape: [n_tasks]
        delta: Significance level of statistical test.
        use_hsic: use HSIC? If False, Levene is used. 
        return_n_best: number of subsets to return. 
    """

    train_x, train_y, valid_x, valid_y, n_ex_train, n_ex_valid = \
      utils.split_train_valid(x, y, n_ex, valid_split)
    
    subset = full_search(train_x, train_y, valid_x, valid_y,
                         n_ex_train, n_ex_valid, use_hsic, 
                         delta, return_n_best = return_n_best)

    return subset


def greedy_subset(x, y, n_ex, delta, valid_split, use_hsic = False):

    """
    Run Algorithm 2 for greedy subset search. 

    Args:
        x: train features. Shape [n_examples, n_features].
        y: train labels. Shape [n_examples, 1].
        n_ex: list with number of examples per task (should be ordered in 
          train_x and train_y). Shape: [n_tasks]
        delta: Significance level of statistical test.
        use_hsic: use HSIC? If False, Levene is used. 
        return_n_best: number of subsets to return. 
    """
    train_x, train_y, valid_x, valid_y, n_ex_train, n_ex_valid = \
      utils.split_train_valid(x, y, n_ex, valid_split)
    subset = greedy_search(train_x, train_y, valid_x, valid_y, n_ex_train, 
                           n_ex_valid, use_hsic, delta)

    return np.array(subset)
