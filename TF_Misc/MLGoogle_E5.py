# numpy slicing
#   nparray[:,0]  # return array consisting of only col 0
#   nparray[:2, 1:] # return row 0 and 1, starting with column 1
#   nparray[2:,1:]  # return row 2->, starting with column 1
#   nparray[:,1,] # return everything starting with col 1 (i.e. rm col 0)
#   nparray[2,:]  # return 3rd row
# operations (on numpy arrays)
#   dot(v1, v2)
#   unique(y)   # returns number of unique entries in array
#   copy()      # copies array
#   zeros(10)   # creates array of size 10 and initializes to 0
#   ones(10)    # ...
#   v1.shape    # Returns [ rows, cols ]

import shutil # shutil.copytree
import os     # os.path.exists, os.path.join
from numpy import array
from numpy import concatenate
from numpy import dot
from numpy import loadtxt
from numpy import unique
from numpy import zeros
from math import exp
from math import log
import random
import sys

def normalizer(w, x):
    d = len(x)
    c = len(w) / d
    Z = 0.0
    for j in range(c):
        dotp = dot( w[j*d:(j+1)*d], x )
        Z += exp( dotp )
    return Z

def ConditionalProbability(w, x, y):
    d = len(x)
    c = len(w) / d
    w_y = w[d * y : d * (y+1)]
    norm = normalizer(w, x)
    return exp(dot(w_y, x)) / norm

def TestConditionalProbability():
    print "-------- TestConditionalProbability"
    x_1 = array([1, 0])
    x_2 = array([0, 1])
    y_1 = 0
    y_2 = 1
    w = array([5, 0, 0, 5])
    # print "shape of w: %s" % w.shape
    print "Pr[y_1 | x_1] = %f" % ConditionalProbability(w, x_1, y_1)
    print "Pr[y_2 | x_2] = %f" % ConditionalProbability(w, x_2, y_2)
    print "Pr[y_2 | x_1] = %f" % ConditionalProbability(w, x_1, y_2)
    print "Pr[y_1 | x_2] = %f" % ConditionalProbability(w, x_2, y_1)

def Objective(w, X, y, lam):
    m = X.shape[0] # Number of rows X.shape == [rows, cols]
    d = X.shape[1] # Number of cols
    num_classes = len(w) / d
    objective = 0.0
    for i in range(m):
        Z = normalizer(w, X[i,:])
        objective += log(Z) - dot(w[d*y[i]: d*(y[i]+1)], X[i,:])
    objective += lam * m/2.0 * dot(w, w)
    return objective

def TestObjective():
    print "-------- Test objective function"
    w = array([5, 0, 0, 5])
    x_1 = array([1, 0])
    x_2 = array([0, 1])
    y_1 = 0
    y_2 = 1
    X = array([x_1, x_2])
    y = array([y_1, y_2])
    y_bad = array([y_2, y_1])
    print "Objective good: %s" % Objective(w, X, y, 0.1)
    print "Objective bad:  %s" % Objective(w, X, y_bad, 0.1)

def Gradient(w, x, y, lam):
    Z = normalizer(w, x)
    grad_w = zeros(len(w))
    d = len(x)
    num_classes = len(w) / d
    for i in range(num_classes):
        w_i = w[d * i: d * (i+1)]
        grad_wi = (exp(dot(w_i, x)) / Z - int(y == i)) * x + lam*w_i
        grad_w[d*i: d*(i+1)] = grad_wi.copy()
    return grad_w

def TestGradient():
    print "-------- TestGradient"
    w = array([5, 0, 0, 5])
    x_1 = array([1, 0])
    x_2 = array([0, 1])
    y_1 = 0
    y_2 = 1
    g_good = Gradient(w, x_1, y_1, 0.0)
    g_bad  = Gradient(w, x_1, y_2, 0.0)
    print '||g_good||^2 = %f' % dot(g_good, g_good)
    print '||g_bad||^2  = %f' % dot(g_bad,  g_bad)

def LogisticRegressionTrain(X, y, lam, max_rounds):
    random.seed(0)
    const_rate = 0.01
    num_classes = len(unique(y))
    m = X.shape[0]
    d = X.shape[1]
    w = zeros(num_classes * d)
    for t in range(max_rounds):
        i = int(random.random() * m)
        grad = Gradient(w, X[i,:], y[i], lam)
        w = w - const_rate*grad
        if t % 1000 == 0:
            obj = Objective(w, X, y, lam)
            sys.stdout.write("\r iter:%d, obj:%f\n" % (t, obj))
    return w

def TestLogisticRegressionTrain():
    datadir = '/Users/magnushyttsten/Documents/python/mnist'
    if not os.path.exists(datadir):
        print "Could not find mnist at: ", datadir
        # to copy would be: shutil.copytree(src_dir, target_dir)
        return
    trainfile = os.path.join(datadir, 'train.csv')
    data = loadtxt(open(trainfile, "rb"), dtype=int, delimiter=",", skiprows=1)
    data_size = data.shape[0]
    num_train = data_size * 0.8
    
    y_train = data[:num_train, 0]
    X_train = data[:num_train, 1:] / 255.0 # normalize to lie in [0,1]
    y_test  = data[num_train:, 0]
    X_test  = data[num_train:, 1:] / 255.0
    print 'num_train: %d, num_test: %d' % (len(y_train), len(y_test))
    w = LogisticRegressionTrain(X_train, y_train, 0.005, 50000)

    pred = Predict(w, X_test)
    diff = pred - y_test
    print "test accuracy:  %f" % (float(sum(diff == 0)) / len(diff))

    pred = Predict(w, X_train)
    diff = pred - y_train
    print "train accuracy: %f" % (float(sum(diff==0)) / len(diff))

    return w

def Predict(w, X):
    m = X.shape[0]
    d = X.shape[1]
    num_classes = len(w) / d
    result = zeros(m)
    for i in range(m):
        x_i = X[i]
        index = 0
        best_prob = 0.0
        best_class = -1
        for j in range(num_classes):
            current_prob = ConditionalProbability(w, X[i,:], j)
            if current_prob > best_prob:
                best_prob = current_prob
                best_class = j
        result[i] = best_class
    return result

# TestConditionalProbability()
# TestObjective()
# TestGradient()

TestLogisticRegressionTrain()
