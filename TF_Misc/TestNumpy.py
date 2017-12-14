
print "***********************************************************************"
print "***********************************************************************"
print "***********************************************************************"
print "Numpy tests\n";

import numpy as np

from numpy import arange
from numpy import array
from numpy import concatenate
from numpy import copy
from numpy import dot
from numpy import ix_
from numpy import loadtxt
from numpy import logical_and
from numpy import nonzero
from numpy import shape
from numpy import tile
from numpy import unique
from numpy import zeros
from math import exp
from math import log

print "**********************"
# Syntax a[rowStarIncl:rowEndExcl, colStartIncl:colEndExcl]
a = array([
    [0,  1,  2,  3],
    [10, 11, 12, 13],
    [20, 21, 22, 23]])
print "Slicing"
print "--- [0,  10, 20] = slice column"
print a[:,0]    
print "--- [20, 21, 22, 23] = slice row"
print a[2,:]
print "--- [[1, 2, 3], [11, 12, 13]] = row 0 and 1, starting with col 1"
print a[:2, 1:]
print "---  [[1, 2], [11, 12]] "
print a[0:2, 1:3]
print "---  a[1:3] "
print a[1:3]

print "**********************"
print "Operation"

print "--- Shape"
print "   shape(a): ", shape(a)
myshape1 = array([1, 2, 3])
myshape2 = array([[1, 2, 3], [4, 5, 6]])
print "   shape1:    ", shape(myshape1)
print "   shape2:    ", shape(myshape2)

print "--- Dot operation"
mydota = array([[1,2],[3,5]])
mydotb = array([[7,11],[13,17]])
mydotc = array([[1,2],[3,4]])
mydotd = array([[10,20],[100,200]])
print dot(mydota, mydotb)
print dot(mydotc, mydotd)

print "--- Unique"
print unique(array([1, 1, 2, 2, 3]))
print unique(array([[1, 1], [1, 2], [1, 1]]))

print "--- Copy"
print copy(array([1, 1, 2, 2, 3]))

print "--- Zeroes..."
print zeros(3) # [.1, .1, .1]
print zeros(3, dtype=np.int)
print zeros((2, 1))
mydim = (2, 2)
print zeros(mydim)
print "Also empyt, eye(diag. ones), identity, ones, full(fill)"

print "--- arange"
myarange = arange(5)
print myarange

print "--- reshape"
print arange(10).reshape(2, 5)

print "--- logical_and >1, < 4"
mylogicaland = logical_and(myarange > 1, myarange < 4)
print mylogicaland
print shape(mylogicaland)

print "--- get matching indexes that are TRUE (non_zero)"
mynonzero = nonzero(mylogicaland)
print "   result: ", mynonzero
print "   result[0]: ", mynonzero[0]
print "   shape(result): ", shape(mynonzero)
print "   shape(result[0]): ", shape(mynonzero[0])

print "--- get by indices array: THIS IS COOL"
# This is make possible by numpy (not a feature of python)
mygetbyindices = array([[0,1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14]])
print "   result: ", mygetbyindices[mynonzero, [2,1]]

print "--- ix_ mesh"
# a[np.ix_([1,3],[2,5])] returns the array 
#    --> [[a[1,2] a[1,5]], [a[3,2] a[3,5]]]
myix = array([
    [10,11,12,13,14,15],
    [20,21,22,23,24,25],
    [30,31,32,33,34,35],
    [40,41,42,43,44,45],
    [50,51,52,53,54,55],
    [60,61,62,63,64,65]])
myixargNot = array([[1,3], [2,5]])
myixarg = ix_([1,3], [2,5])
print "    myixargNot:\n", myixargNot
print "    myixarg:\n", myixarg
print "    myisargNot.shape: ", myixargNot.shape
print "    myisarg[0].shape: ", myixarg[0].shape
print "      myisarg[0][0].shape: ", myixarg[0][0].shape
print "        myisarg[0][0][0]: ", myixarg[0][0][0]
print "      myisarg[0][1].shape: ", myixarg[0][1].shape
print "        myisarg[0][1][0]: ", myixarg[0][1][0]
print "    myisarg[1].shape: ", myixarg[1].shape
print "      myisarg[1][0].shape: ", myixarg[1].shape
print "        myisarg[1][0][0]: ", myixarg[1][0][0]
print "        myisarg[1][0][1]: ", myixarg[1][0][1]
print "    myixres0:\n", myix[myixarg]
print "    myixres1:\n", myix[ix_([1,3],[2,5])]
print "    myixres2:\n", myix[[1,3],[2,5]]
print "    myixres3:\n", myix[[[1],[3]],[2,5]]
print "    myixres4:\n", myix[array([[1],[3]]), array([[2,5]])]

print "--- tile (repeat items (x number of rows, y number of times))"
print tile([1,2,3], 2)
print tile([1,2,3], (1, 2))
print tile([1,2,3], (2,3))
print tile([1,2,3], (2,1))

print "--- Understand tiling and transposing for L7"
print "    Observe that l7r creates one row with result as columns"
print "    Duplicate that row and transpose to give same rows as 3p and blocks"
print "    And two columns of #games played to match 3p and blocks vector div"
l7stats = array([[0,1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14]])
l7indices = array([1, 2, 4])
l7r =  l7stats[l7indices, 0]
print "Result\n", l7r
l7rTiled = tile(l7r, (2,1))
print "Result tiled\n", l7rTiled
l7rTiledT = l7rTiled.T
print "Result tiled transpose\n", l7rTiledT

print "\n--- Mean and standard deviation"
mymean = array([
    [4,5,6], 
    [1,2,3], 
    [5,6,8],
    [7,8,9]])
print "   Mean(0,calculate mean of each col): ", mymean.mean(axis=0)
print "   Mean(1,calculate mean of each row): ", mymean.mean(axis=1)
mysd1 = array([
    [1,2,3], 
    [1,2,3], 
    [1,2,3],
    [1,2,3]])
mysd2 = array([
    [1,1,1], 
    [2,2,2], 
    [3,3,3],
    [4,4,4]])
print "   Stddev1(0, over cols): ", mysd1.std(axis=0)
print "   Stddev1(1, over rows): ", mysd1.std(axis=1)
print "   Stddev2(0, over cols): ", mysd2.std(axis=0)
print "   Stddev2(1, over rows): ", mysd2.std(axis=1)

# Mathlabs plot function
# pyplot.scatter(array([10,11,12]), array([20,21,22]))
# pyplot.xlabel("X-label")
# pyplot.ylabel("Y-label")

print "***********************************************************************"
print "***********************************************************************"
print "***********************************************************************"
print "Python tests\n";
'''
print "--- Zip and asterisk"
myzipx = [[100,101],[110,111],[120,121],[130,131],[140,141]]
myzipy = [20,21,22,23,24,25,26]
myzipRawCombine = zip(myzipx,myzipy)
print "Raw zip: ", myzipRawCombine
for myzipxe,myzipye in zip(myzipx, myzipy):
    print "(x:%s, y:%d)" % (myzipxe, myzipye)
myzipl = [x for x,y in zip(myzipx,myzipy) if y>20 and y<25]
print "zip list result: ", myzipl
myzipxe,myzipye = zip(*myzipl)
print "X is", myzipxe
print "Y is", myzipye

print "\n--- String join"
import string
mystring = ['Alpha', 'Bravo', 'Charlie', 'Delta', 'Echo', 'Foxtrot']
print "Here is unzip: ", zip(mystring)
print "Here is join:  ", string.join(mystring)
'''

print "**********************"
print "Loading a file"
'''
import shutil, os
myloadfile = os.path.join('/tmp', 'test.csv')
myloadfiledata = np.loadtxt(open(myloadfile,"rb"), dtype=int, delimiter=",", skiprows=1)
print "Filedata\n"
print myloadfiledata
print "\n\n"
'''

print "**********************"
print "random"
from numpy import random
random.seed(0)
myrandom_examples = 10
myrandom_series = 2 * (random.random([myrandom_examples, 2]) - 0.5)
print "   shape: ", myrandom_series.shape
print "   series: ", myrandom_series

