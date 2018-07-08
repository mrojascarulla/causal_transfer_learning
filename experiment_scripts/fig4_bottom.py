#!/usr/bin/env python
import sys
sys.path.append('../code')
import numpy as np
from sklearn import linear_model
import argparse

import data
import subset_search
import utils
import plotting
import mSDA
import timeit

import cPickle as pickle
import os

np.random.seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', default = '../results')
parser.add_argument('--n_task', default=13)
parser.add_argument('--merge_dica', default=0)
parser.add_argument('--n', default=1000)
parser.add_argument('--p', default = 30)
parser.add_argument('--p_s', default = 15)
parser.add_argument('--p_conf', default = 0)
parser.add_argument('--eps', default = 6)
parser.add_argument('--g', default = 1)
parser.add_argument('--lambd', default = 0.5)
parser.add_argument('--lambd_test', default = 0.99)
parser.add_argument('--use_hsic', default = 0)
parser.add_argument('--alpha_test', default = 0.05)
parser.add_argument('--n_repeat', default = 100)
parser.add_argument('--max_l', default = 100)
parser.add_argument('--n_ul', default = 100)
args = parser.parse_args()

save_dir = args.save_dir

if not os.path.exists(save_dir):
  os.makedirs(save_dir)

save_dir = os.path.join(save_dir, 'fig4_bottom')
if not os.path.exists(save_dir):
  os.makedirs(save_dir)

#If only plot is true, loads results, merges with DICA and plots again. 
if int(args.merge_dica) == 1:
    utils.merge_results('results.pkl', 'dica.pkl', 'dica', save_dir)
    plotting.plot_tl(os.path.join(save_dir, 'merged.pkl'))
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

dataset = data.gauss_tl(n_task, n, p, p_s, p_conf, eps ,g, lambd, lambd_test)
x_train = dataset.train['x_train']
y_train = dataset.train['y_train']
n_ex = dataset.train['n_ex']

lr = linear_model.LinearRegression()
lr.fit(x_train, y_train)

# Define test
x_test = dataset.test['x_test']
y_test = dataset.test['y_test']

n_train_tasks = np.arange(2,n_task)
n_repeat = int(args.n_repeat)
true_s = np.arange(p_s)

results = {}
methods = [
            'pool',
            'sgreed',
            'strue',
            'mean',
            'msda'
          ]
n_train_tasks = np.arange(n_train_tasks[0], n_train_tasks[-1], 1)
color_dict, markers, legends = utils.get_color_dict()

for m in methods:
  results[m]  = np.zeros((n_repeat, n_train_tasks.size))

for rep in xrange(n_repeat):
  print rep
  x_train, y_train = dataset.resample(n_task, n)

  x_test = dataset.test['x_test']
  y_test = dataset.test['y_test']

  for index, t in np.ndenumerate(n_train_tasks):
    x_temp = x_train[0:np.cumsum(n_ex)[t], :]
    y_temp = y_train[0:np.cumsum(n_ex)[t], :]

    s_greedy = subset_search.greedy_subset(x_temp, y_temp, n_ex[0:t], 
                                           delta=alpha_test, 
                                           valid_split=0.6,
                                           use_hsic=use_hsic)
    # Pooled
    lr_temp = linear_model.LinearRegression()
    lr_temp.fit(x_temp, y_temp)
    results['pool'][rep, index] = utils.mse(lr_temp, x_test, y_test)

    # Mean
    error_mean = np.mean((y_test - np.mean(y_temp))**2)
    results['mean'][rep, index] = error_mean


    # Estimated greedy S
    lr_sg_temp = linear_model.LinearRegression()
    lr_sg_temp.fit(x_temp[:,s_greedy], y_temp)
    results['sgreed'][rep, index] = utils.mse(lr_sg_temp, x_test[:,s_greedy], y_test)

    # True S
    lr_true_temp = linear_model.LinearRegression()
    lr_true_temp.fit(x_temp[:,true_s], y_temp)
    results['strue'][rep, index] = utils.mse(lr_true_temp,x_test[:,true_s], y_test)

    #mSDA
    p = np.linspace(0,1,10)
    p_cv = mSDA.mSDA_cv(p, x_temp, y_temp, n_cv = t)
    fit_sda = mSDA.mSDA(x_temp.T,p_cv,1)
    x_sda = fit_sda[-1][-1].T
    w_sda = fit_sda[0]
    x_test_sda = mSDA.mSDA_features(w_sda, x_test.T).T

    lr_sda = linear_model.LinearRegression()
    lr_sda.fit(x_sda, y_temp)
    results['msda'][rep, index] = utils.mse(lr_sda, x_test_sda, y_test)


save_all = {}
save_all['results'] = results
save_all['plotting'] = [methods, color_dict, legends, markers]
save_all['n_train_tasks'] = n_train_tasks

file_name = ['tl_norm_', str(n_repeat), str(eps), str(g)]
file_name = '_'.join(file_name)

with open(os.path.join(save_dir, file_name+'.pkl'),'wb') as f:
  pickle.dump(save_all, f)

#Create plot
plotting.plot_tl(os.path.join(save_dir, file_name + '.pkl'))

