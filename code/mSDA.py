from numpy import *
import numpy as np
import collections, re
import sys
from argparse import ArgumentParser
from sklearn.model_selection import KFold
from sklearn import linear_model
import utils

def mDA(X, p):
    X = vstack((X,ones((1,shape(X)[1]))))
    d = shape(X)[0]
    q = vstack((ones((d-1,1))*(1-p),1))
    S = dot(X,X.transpose())
    Q = S * dot(q,q.transpose())
    for i in range(d):
        Q[i,i] = q[i,0]*S[i,i]
    qrep = q.transpose()
    for j in range(d-1):
        qrep = vstack((qrep,q.transpose()))
    P = S * qrep
    W = linalg.solve((Q + 10**(-5)*eye(d)).transpose(), P[:d-1].transpose()).transpose()

    h = (dot(W,X))
    return (W,h)

def mSDA(X, p, l):
    d,n = shape(X)
    Ws = zeros((l,d,d+1))
    hs = zeros((l+1,d,n))
    hs[0] = X 
    for t in range(l):
        Ws[t], hs[t+1] = mDA(hs[t], p)
    return (Ws, hs)

def mSDA_features(w, x):

  for i in xrange(w.shape[0]):
    x = append(ones((1,x.shape[1])), x,0)
    x = dot(w[i], x)

  return x

def mSDA_cv(p, x, y, n_cv=5):

  kf = KFold(n_splits = n_cv)
  res = np.zeros((p.size, n_cv))

  for j, pj in enumerate(p):
    i = 0
    for train, test in kf.split(x):
      x_temp, y_temp = x[train], y[train]
      x_test, y_test = x[test], y[test]

      fit_sda = mSDA(x_temp.T,pj,1)
      x_sda = fit_sda[-1][-1].T
      w_sda = fit_sda[0]
      x_test_sda = mSDA_features(w_sda, x_test.T).T

      lr_sda = linear_model.LinearRegression()
      lr_sda.fit(x_sda, y_temp)
      res[j,i] = utils.mse(lr_sda, x_test_sda, y_test)

      i += 1
  res = np.mean(res,1)
  return  p[np.argmin(res)]

"""


def parseArgs():
    parser = ArgumentParser(description='mSDA implementation for python')
    parser.add_argument('f', help='The input file in LIBSVM format', metavar='FILE')
    parser.add_argument('p', help='The probability of corruption', metavar='P', type=float)
    parser.add_argument('l', help='The number of layers', metavar='L',type=int)
    parser.add_argument('-d', '--dimension', help='The dimensionality of the data', dest='d',
                        metavar='DIM', type=int, default = -1)
    parser.add_argument('-i', '--input', help='Include input in output files', action = 'store_true',
                        dest='i', default = False)
    parser.add_argument('-a', '--allLayers', help='Include all intermediate layers in output files', 
                        action = 'store_true', dest='a', default = False)
    parser.add_argument('-o', '--output', help='The name of the output file. Default: out.txt', 
                        metavar='OUT', dest='o', type=str, default='out.txt')
    parser.add_argument('-t', '--testFile', metavar='TEST', dest='t', type=str, default='', 
                        help='File containing test data in LIBSVM format. Output will be written to a separate file named test_OUT')
    return parser.parse_args()


#Parse a file in LIBSVM format and return bag-of-words vectors.
def parseFile(fileName):
    f = open(fileName)
    d = args.d
    #Find the dimentionality of the data if it is not provided.
    if (d == -1):
        for line in f:
            words = line.split()[1:]
            di = max([int(word.split(':')[0]) for word in words])
            d = max(d,di)
        f.seek(0)
    #Parse the data into bag-of-words vectors.
    X = zeros((d,0))
    labels = []
    for line in f:
        words = line.split()
        labels.append(words[0])
        x = zeros((d,1))
        for word in words[1:]:
            i,w = word.split(':')
            if int(i) > d:
                sys.exit('d')
            x[int(i) - 1][0] = int(w)
        X = hstack((X,matrix(x)))
    f.close()
    return (X, labels)


#Write bag-of-words vectors to a file in LIBSVM format.
def writeToFile(fileName, Hs, labels):
    x,y,z = shape(Hs)
    out = open(fileName, 'w')
    for i in range(x):
        if (i == 0 and args.i) or (i > 0 and args.a) or i == (x-1):
            for j in range(z):
                line = labels[j] + ' '
                for k in range(y):
                    if Hs[i][k][j] != 0:
                        line += str(k+1) + ':' + str(Hs[i][k][j]) + ' '
                line += '\n'
                out.write(line)
    out.close()

    



errors = {'l':'Error: The number of layers must be positive.',
          'p':'Error: The probability of corruption must be between 0 and 1.',
          'fileName':'Error: invalid file name',
          'fileFormat':'Error: invalid file format',
          'd':'Error: given dimension is incorrect'}


args = parseArgs()

try:
    if args.l < 0:
        sys.exit('l')
    if args.p < 0 or args.p > 1:
        sys.exit('p')

    (X, labels) = parseFile(args.f)
    (Ws, Hs) = mSDA(X, args.p, args.l)
    writeToFile(args.o, Hs, labels)
    
    #Transform and write out test data, if any.
    if (args.t != ''):
        (tX, tlabels) = parseFile(args.t)
        d,n = shape(tX)
        tHs = zeros((args.l+1,d,n))
        tHs[0] = tX
        for t in range(args.l):
            tHs[t+1] = tanh(dot(Ws[t],vstack((tHs[t],ones((1,n))))))
        writeToFile('test_' + args.o, tHs, tlabels)
    
except IOError:
    print(errors['fileName'])
except SystemExit as e:
    print(errors[e[0]])
except:
    print(errors['fileFormat'])
"""
