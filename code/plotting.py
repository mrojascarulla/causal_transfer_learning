#!/usr/bin/env python
# Copyright (c) 2014 - 2018  Mateo Rojas-Carulla  [mrojascarulla@gmail.com]
# All rights reserved.  See the file COPYING for license terms.

import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib import rc

import data
import subset_search
import utils
import cPickle as pickle

import sys, os


def plot_tl(file_name, ylim=None):

  with open(file_name,'rb') as f:
    save_all = pickle.load(f)

  name = file_name.split('_')[-2:]

  methods = save_all['plotting'][0]
  color_dict = save_all['plotting'][1]
  legends = save_all['plotting'][2]
  markers = save_all['plotting'][3]
  markers['strue'] = '*'
  results = save_all['results']
  n_train_tasks = save_all['n_train_tasks']
  n_repeat = results[methods[0]].shape[0]

  fig, ax = plt.subplots(1)
  offset = 0.0
  for i, m in enumerate(methods):
    ax.plot(n_train_tasks + i*offset, (np.mean(np.log(results[m]), 0)), c = color_dict[m],
            alpha = 0.5)
    ax.errorbar(n_train_tasks + i*offset, (np.mean(np.log(results[m]), 0)), 
                                  yerr = np.std(np.log(results[m]),0)*1.96/np.sqrt(n_repeat),
                                  c = color_dict[m], label = legends[m], fmt = markers[m], 
                                  alpha = 0.5)

  ax.set_xlabel(r'# of training tasks $T$', fontsize = 15)
  ax.set_ylabel(r'$\log \widehat{\mathrm{MSE}}$', fontsize = 15)
  ax.set_xticks(n_train_tasks)
  xticks = [r"$"+str(t)+"$" for t in n_train_tasks]
  ax.set_xticklabels(xticks, fontsize = 15)
  ax.grid(True, axis = 'y', linestyle = ':')
  if ylim:
      plt.ylim([1,ylim])
  lgd = ax.legend(prop={'size':15}, bbox_to_anchor=(0.,1.08,1.,0.102), ncol = 4, loc = 'center',
                            borderaxespad = 0)

  save_dir = '/'.join(file_name.split('/')[0:-1])
  plt.savefig(os.path.join(save_dir, 'plot.pdf'),
              bbox_extra_artists=(lgd,), 
              bbox_inches='tight',
              format = 'pdf', dpi = 1000)

def plot_mtl_mse(file_name):

  with open(file_name,'rb') as f:
    save_all = pickle.load(f)

  name = file_name.split('_')[-2:]
  methods = save_all['plotting'][0]
  color_dict = save_all['plotting'][1]
  legends = save_all['plotting'][2]
  markers = save_all['plotting'][3]
  markers['strue'] = '*'
  results = save_all['results']
  samp_array = save_all['n_samp']
  n_repeat = results[methods[0]].shape[0]

  fig, ax = plt.subplots(1)
  offset = 0.0
  
  n_dim = len(results['dom'].shape)

  for i, m in enumerate(methods):
    if n_dim == 3:
      y = np.mean(np.log(results[m]), (0,-1))
      y_std = np.std(np.log(results[m]), (0,-1))
    else:
      y = np.mean(np.log(results[m]),0)
      y_std = np.std(np.log(results[m]),0)

    ax.plot(samp_array + i*offset, y, c = color_dict[m],
            alpha = 0.5, linestyle='-', label = legends[m], marker=markers[m])
    ax.errorbar(samp_array + i*offset, y, 
                                  yerr = y_std*1.96/np.sqrt(n_repeat),
                                  c = color_dict[m], fmt = markers[m], 
                                  alpha = 0.5)

  ax.set_xlabel(r'# of training tasks $T$', fontsize = 15)
  ax.set_ylabel(r'$\log \widehat{\mathrm{MSE}}$', fontsize = 15)
  ax.set_xticks(np.arange(100,1000,200))
  xticks = [r"$"+str(t)+"$" for t in np.arange(100,1000,200)]
  ax.set_xticklabels(xticks, fontsize = 12)
  ax.grid(True, axis = 'y', linestyle = ':')
  lgd = ax.legend(prop={'size':15}, bbox_to_anchor=(0.,1.08,1.,0.102), 
                  ncol = 3, loc = 'center',
                  borderaxespad = 0)

  save_dir = '/'.join(file_name.split('/')[0:-1])
  plt.savefig(os.path.join(save_dir, 'plot_mse.pdf'),bbox_extra_artists=(lgd,), 
              bbox_inches='tight',
              format = 'pdf', dpi = 1000)


def plot_deltas(file_name):

  with open(file_name,'rb') as f:
    save_all = pickle.load(f)

  name = file_name.split('_')[-2:]

  methods = save_all['plotting'][0]
  color_dict = save_all['plotting'][1]
  legends = save_all['plotting'][2]
  markers = save_all['plotting'][3]
  markers['strue'] = '*'
  results = save_all['results']
  deltas = save_all['deltas']
  n_repeat = results[methods[0]].shape[0]

  if 'msda' in methods:
    methods.remove('msda')

  fig, ax = plt.subplots(1)
  offset = 0.0
  for i, m in enumerate(methods):
    
    if m == 'pool' or m == 'strue':
      xaxis = np.arange(-0.01, deltas[-1]+0.01, 0.001)
      dashed_mean = np.array(xaxis.size*[np.mean(np.log(results[m]), 0)[0]])
      dashed_std = np.array(xaxis.size*[1.96/np.sqrt(n_repeat)*np.std(np.log(results[m]),0)[0]])

      ax.plot(xaxis, dashed_mean,c = color_dict[m], linestyle = '--', label = legends[m])
      ax.fill_between(xaxis, dashed_mean - dashed_std, dashed_mean + dashed_std, alpha = 0.2,
                      facecolor = color_dict[m])
    elif m == 'mean':
      continue
    else:
      ax.plot(deltas + i*offset, (np.mean(np.log(results[m]), 0)), c = color_dict[m],
              alpha = 0.5)
      ax.errorbar(deltas + i*offset, (np.mean(np.log(results[m]), 0)), 
                                    yerr = np.std(np.log(results[m]),0)*1.96/np.sqrt(n_repeat),
                                    c = color_dict[m], label = legends[m], fmt = markers[m], 
                                    alpha = 0.5)

  ax.set_xlabel(r'Acceptance level $\delta$', fontsize = 15)
  ax.set_ylabel(r'$\widehat{\mathrm{MSE}}$', fontsize = 15)

  deltas_sub = np.array([deltas[s] for s in np.arange(0, deltas.size, 3)])
  plt.xticks(deltas_sub, fontsize = 10)
  ax.grid(True, axis = 'y', linestyle = ':')
  ax.set_xlim([-0.01, deltas[-1]+0.005])
  ax.set_ylim(bottom = 1.2)
  lgd = ax.legend(prop={'size':13}, bbox_to_anchor=(0.,1.02,1.,0.102), ncol = 4, loc = 'center',
                            borderaxespad = 0)

  save_dir = '/'.join(file_name.split('/')[0:-1])
  plt.savefig(os.path.join(save_dir, 'plot.pdf'),bbox_extra_artists=(lgd,), 
              bbox_inches='tight',
              format = 'pdf', dpi = 1000)

def plot_mtl(file_name):
    
  with open(file_name,'rb') as f:
    save_all = pickle.load(f)

  methods = save_all['plotting'][0]
  color_dict = save_all['plotting'][1]
  legends = save_all['plotting'][2]
  markers = save_all['plotting'][3]
  results = save_all['results']
  samp_array = save_all['n_samp']
  
  n_dim = len(results['dom'].shape)
  if n_dim == 3:
    dom_av_tasks = np.sum(results['dom'],-1)
  else:
    dom_av_tasks = results['dom']

  legends['shat'] = '$\\beta^{CS(\hat S +)}$'
  legends['strue'] = '$\\beta^{CS(cau +)}$'
  n_repeat = results['dom'].shape[0]
  fig, ax = plt.subplots(1)
  offset = 0

  for i, m in enumerate(methods):
    if m == 'dom': continue
    if n_dim == 3:
      av_tasks_method = np.sum(results[m],-1)
    else:
      av_tasks_method = results[m]
   
    comp = np.sum((dom_av_tasks - av_tasks_method)>0, 0).astype(int)/float(n_repeat)
    ax.plot(samp_array + i*offset, comp, 
             c = color_dict[m],
             label = legends[m],
             marker = markers[m],
             markersize = 5,
             linestyle = '-', 
             alpha = 0.5)

  up = samp_array[-1]+10
  base = np.ones(up)
              
  plt.fill_between(np.arange(up),0.4*base, 0.6*base, 
                  alpha=0.35, edgecolor='#b3b3cc', facecolor='#b3b3cc', interpolate = True)
  plt.plot(np.arange(up), up*[0.5], color='grey',ls='--')

   
  plt.xlabel(r'Number of labeled examples in $T$', fontsize = 15)
  plt.ylabel(r'Percentage of simulations in' '\n' r'which $\beta^{dom}$ is outperformed', fontsize = 15)

  lgd = ax.legend(prop = {'size': 15}, 
                  bbox_to_anchor = (0., 1.08, 1., 0.102),
                  ncol = 3,
                  loc = 'center',
                  borderaxespad = 0)

  plt.xlim([0,samp_array[-1]+10])
  textstr = r'Reference: $\beta^{dom}$'

  # these are matplotlib.patch.Patch properties
  props = dict(boxstyle='round', facecolor='grey', alpha=0.3)
  x_coord = samp_array[samp_array.size/2-2]
  plt.text(x_coord,0.15,textstr, fontsize = 20, bbox = props)
 
  save_dir = '/'.join(file_name.split('/')[0:-1])
  plt.savefig(os.path.join(save_dir, 'plot.pdf'), 
              bbox_extra_artistics=(lgd,),
              bbox_inches = 'tight',
              format = 'pdf',
              dpi = 1000)
  
def plot_mtl_subset(file_name):
    
  with open(file_name,'rb') as f:
    save_all = pickle.load(f)

  methods = save_all['plotting'][0]
  color_dict = save_all['plotting'][1]
  legends = save_all['plotting'][2]
  markers = save_all['plotting'][3]
  results = save_all['results']
  subsize = np.append(save_all['subsize'],6)
  
  n_dim = len(results['dom'].shape)

  if n_dim == 3:
    dom_av_tasks = np.sum(results['dom'],-1)
  else:
    dom_av_tasks = results['dom']

  n_repeat = results['dom'].shape[0]
  fig, ax = plt.subplots(1)
  offset = 0

  for i, m in enumerate(methods):
    if m == 'dom': continue
    if n_dim == 3:
      av_tasks_method = np.sum(results[m],-1)
    else:
      av_tasks_method = results[m]

    comp = np.sum((dom_av_tasks - av_tasks_method)>0, 0).astype(int)/float(n_repeat)
    ax.plot(subsize + i*offset, comp, 
             c = color_dict[m],
             label = legends[m],
             marker = markers[m],
             markersize = 5,
             linestyle = '-', 
             alpha = 0.2)

  up = subsize[-1]+3
  base = np.ones(up)
              
  plt.fill_between(np.arange(up),0.4*base, 0.6*base, 
                  alpha=0.35, edgecolor='#b3b3cc', facecolor='#b3b3cc', interpolate = True)
  plt.plot(np.arange(up), up*[0.5], color='grey',ls='--')

   
  plt.xlabel(r'Size of $S^*$', fontsize = 15)
  plt.ylabel(r'Percentage of simulations in' '\n' r'which $\beta^{dom}$ is outperformed', fontsize = 15)

  lgd = ax.legend(prop = {'size': 15}, 
                  bbox_to_anchor = (0., 1.08, 1., 0.102),
                  ncol = 3,
                  loc = 'center',
                  borderaxespad = 0)

  plt.xlim([0.5,6.5])
  textstr = r'Reference: $\beta^{dom}$'

  # these are matplotlib.patch.Patch properties
  props = dict(boxstyle='round', facecolor='grey', alpha=0.3)
  x_coord = subsize[subsize.size/2-3]
  plt.text(x_coord,0.15,textstr, fontsize = 20, bbox = props)
 
  pref = file_name.split('/')[0]

  plt.savefig('mtl_subset_'+ pref + '.pdf', 
              bbox_extra_artistics=(lgd,),
              bbox_inches = 'tight',
              format = 'pdf',
              dpi = 1000)
  
  plt.show()



def plot_mtl_ul(file_name):
    
  with open(file_name,'rb') as f:
    save_all = pickle.load(f)

  methods = save_all['plotting'][0]
  color_dict = save_all['plotting'][1]
  legends = save_all['plotting'][2]
  markers = save_all['plotting'][3]
  results = save_all['results']
  samp_array = save_all['n_samp']
  samp_array = samp_array[1:]
  n_dim = len(results['cauul'].shape)

  for m in methods:
    results[m] = results[m][:,1:,:]

  if n_dim == 3:
    cau_av_tasks = np.sum(results['causharp'],-1)
  else:
    cau_av_tasks = results['causharp']

  n_repeat = results['causharp'].shape[0]
  fig, ax = plt.subplots(1)
  offset = 0

  for i, m in enumerate(methods):
    if m == 'causharp': continue
    if n_dim == 3:
      av_tasks_method = np.sum(results[m],-1)
    else:
      av_tasks_method = results[m]

    comp = np.sum((cau_av_tasks - av_tasks_method)>=0, 0).astype(int)/float(n_repeat)
    ax.plot(samp_array + i*offset, comp, 
             c = color_dict[m],
             label = legends[m],
             marker = markers[m],
             markersize = 5,
             linestyle = '-', 
             alpha = 0.5)

  up = samp_array[-1]+10
  base = np.ones(up)
              
  plt.fill_between(np.arange(up),0.4*base, 0.6*base, 
                  alpha=0.35, edgecolor='#b3b3cc', facecolor='#b3b3cc', interpolate = True)
  plt.plot(np.arange(up), up*[0.5], color='grey',ls='--')

   
  plt.xlabel(r'Number of unlabeled examples in $T$', fontsize = 15)
  plt.ylabel(r'Percentage of simulations in' '\n' r'which $\beta^{CS(cau\sharp)}$ is outperformed', fontsize = 15)

  lgd = ax.legend(prop = {'size': 15}, 
                  bbox_to_anchor = (0., 1.08, 1., 0.102),
                  ncol = 3,
                  loc = 'center',
                  borderaxespad = 0)

  plt.xlim([0,samp_array[-1]+10])
  textstr = r'Reference: $\beta^{CS(cau\sharp)}$'

  # these are matplotlib.patch.Patch properties
  props = dict(boxstyle='round', facecolor='grey', alpha=0.3)
  plt.text(300,0.15,textstr, fontsize = 20, bbox = props)
 
  pref = file_name.split('/')[0]

  plt.savefig('mtl_ul_'+ pref + '.pdf', 
              bbox_extra_artistics=(lgd,),
              bbox_inches = 'tight',
              format = 'pdf',
              dpi = 1000)
  
  plt.show()
  
def plot_interv(file_name):

  rc('text', usetex = True)

  save_dir = '/'.join(file_name.split('/')[0:-1])

  with open(file_name, 'rb') as f:
    data = pickle.load(f)
  count = data['count']
  n_repeat = data['n_repeat']
  inter = data['inter']

  width = 0.1

  count = count/float(n_repeat)
  locations = np.arange(count.shape[0])
  fig, ax = plt.subplots()

  for i,l in enumerate(count):
    for p in xrange(count.shape[1]):
      if p-3 in inter[i]:
        col = 'r'
      else:
        col = 'b'

      ax.bar(i + (p-3)*width, count[i,p], width-0.02, color = col, alpha = 0.35)

  ax.set_xticks(np.arange(4))
  ax.set_xticklabels([r'-',r'$4$',r'$4,5$',r'$4,5,6$'], fontsize = 15)
  
  ax.set_yticks(np.arange(0,1.1,0.2))
  ax.set_yticklabels([r"$0$", r"$0.2$", r"$0.4$", r"$0.6$", r"$0.8$", r"$1$"], fontsize = 15)

  plt.ylabel(r"Percentage of repetitions for which the" 
             "\n" 
            r"covariates are included in $\hat S$", 
            fontsize = 20,
            multialignment = 'center')
  plt.xlabel(r'Intervened on covariates', fontsize = 20)

  plt.savefig(os.path.join(save_dir, 'plot.pdf'),
              bbox_inches='tight',
              format = 'pdf', dpi = 1000)


def hist_mtl(file_name):
    
  with open(file_name,'rb') as f:
    save_all = pickle.load(f)

  methods = save_all['plotting'][0]
  color_dict = save_all['plotting'][1]
  legends = save_all['plotting'][2]
  markers = save_all['plotting'][3]
  results = save_all['results']
  samp_array = save_all['n_samp']
  n_dim = len(results['dom'].shape)
  dom_av_tasks = np.mean(results['dom'],1)[:,-1]

  n_repeat = results['dom'].shape[0]
  fig, ax = plt.subplots(1)
  offset = 1

  m = 'causharp'
  av_tasks_method = np.mean(results[m],1)[:,-1]

  dif = dom_av_tasks - av_tasks_method 
  plt.hist(dif, bins = 30, facecolor = 'orange', alpha = 0.5,
            edgecolor = 'black',
            linewidth = 1.2)

  plt.xlabel(r'$\mathcal{E}(\beta^{dom})-\mathcal{E}(\beta^{cau\sharp})$', fontsize = 15)
  plt.ylabel(r'Number of repetitions', fontsize = 15)
  
  plt.savefig('mtl_hist.pdf', 
              format = 'pdf',
              dpi = 1000)
  plt.show()

if __name__ == '__main__':
    save_dir = sys.argv[1]
    tl = int(sys.argv[2])

    if tl == 1:
      fn = plot_tl
    elif tl == 0:
      fn = plot_mtl
    elif tl == 2:
      fn = plot_interv
    elif tl == 3:
      fn = plot_deltas
    elif tl == 4:
      fn = plot_mtl_mse
    elif tl == 5:
      fn = plot_mtl_ul
    elif tl == 6:
      fn = plot_mtl_subset
    elif tl == 7:
      fn = hist_mtl

    for f in os.listdir(save_dir):
      if f.split('.')[-1] != 'pkl':
          continue
      print os.path.join(save_dir, f)
      fn(os.path.join(save_dir, f))

