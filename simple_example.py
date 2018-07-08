#!/usr/bin/env python
# Copyright (c) 2014 - 2018  Mateo Rojas-Carulla  [mrojascarulla@gmail.com]
# All rights reserved.  See the file COPYING for license terms.

import numpy as np
import sys
sys.path.append('code')
import subset_search
from sklearn import linear_model

np.random.seed(12)

n_examples_task = 300
n_tasks = 3
n_test_tasks = 100
n_predictors = 3
n_ex = []

#---------------------------------------------------------------
#Parameters of the SEM
#---------------------------------------------------------------

alpha = np.random.uniform(-1, 2.5, 2)
sigma = 1.5
sx1 = 1
sx2 = 0.1
sx3 = 1

train_x = np.zeros((1, n_predictors))
train_y = np.zeros(1)

use_hsic = True
return_mse = False
delta = 0.05

#---------------------------------------------------------------
#Generate training tasks
#---------------------------------------------------------------

for task in range(n_tasks):
    gamma_task = np.random.uniform(-1, 1)
    x1 = np.random.normal(0, sx1,(n_examples_task, 1))
    x3 = np.random.normal(0, sx3, (n_examples_task,1))
    y = alpha[0] * x1 + alpha[1] * x3 + np.random.normal(
      0, sigma, (n_examples_task, 1))
    x2 = gamma_task*y + np.random.normal(0, sx2, (n_examples_task, 1))

    x_task = np.concatenate([x1, x2, x3],axis = 1)
    train_x = np.append(train_x, x_task, axis = 0)
    train_y = np.append(train_y, y)
    n_ex.append(n_examples_task)

n_ex = np.array(n_ex)
train_x =  train_x[1:, :]
train_y = train_y[1:, np.newaxis]

test_x = np.zeros((1, n_predictors))
test_y = np.zeros(1)

#---------------------------------------------------------------
#Generate test tasks
#---------------------------------------------------------------

for task in range(n_test_tasks):

    gamma_task = np.random.uniform(-1,1)
    x1 = np.random.normal(0,sx1,(n_examples_task,1))
    x3 = np.random.normal(0,sx3,(n_examples_task,1))
    y = alpha[0]*x1 + alpha[1]*x3 + np.random.normal(
      0,sigma,(n_examples_task,1))
    x2 = gamma_task*y + np.random.normal(0,sx2,(n_examples_task,1))

    x_task = np.concatenate([x1, x2, x3],axis = 1)
    test_x = np.append(test_x, x_task, axis = 0)
    test_y = np.append(test_y, y)

test_x = test_x[1:,:]
test_y = test_y[1:,np.newaxis]

#---------------------------------------------------------------
#Estimate subset
#---------------------------------------------------------------

s_hat = subset_search.subset(train_x, train_y, n_ex, valid_split=0.5, 
                             delta=0.05, use_hsic=use_hsic)
s_hat_greedy = subset_search.greedy_subset(train_x, train_y, n_ex,  
                                           valid_split=0.5, 
                                           delta=0.05, 
                                           use_hsic=use_hsic)

print "TRANSFER LEARNING: NO LABELED EX. FROM TEST AT TRAINING TIME"
print
print "The complete subset of predictors:"
print tuple(np.arange(train_x.shape[1]))
print "The true causal set for this problem:"
print (0, 2) 
print "The estimated invariant subset is:"
print tuple(s_hat)
print
print "The estimated invariant subset with a greedy procedure is:"
print tuple(s_hat_greedy)
print

#---------------------------------------------------------------
# Fit a model with all predictors and print test error
#---------------------------------------------------------------
regr_pool = linear_model.LinearRegression()
regr_pool.fit(train_x, train_y)
pred_pool = regr_pool.predict(test_x)
mse_pool = np.mean((test_y - pred_pool) ** 2)
print "MSE in test for pooled: %f" % mse_pool

#---------------------------------------------------------------
# Fit a model using only predictors in Shat and print test error
#---------------------------------------------------------------

if s_hat.size >0:
    regr_subset = linear_model.LinearRegression()
    regr_subset.fit(train_x[:,s_hat], train_y)
    pred_subset = regr_subset.predict(test_x[:, s_hat])
else:
    pred_subset = np.mean(train_y)

mse_subset = np.mean((test_y - pred_subset) ** 2)

print "MSE in test using S_hat: %f" % mse_subset

#---------------------------------------------------------------
# Fit a model using only predictors in Shat_greedy and print test error
#---------------------------------------------------------------
if s_hat_greedy.size >0:
    regr_subset = linear_model.LinearRegression()
    regr_subset.fit(train_x[:, s_hat_greedy], train_y)
    pred_subset = regr_subset.predict(test_x[:, s_hat_greedy])
else:
    pred_subset = np.mean(train_y)

mse_subset = np.mean((test_y - pred_subset) ** 2)

print "MSE in test using greedy S_hat: %f" % mse_subset
