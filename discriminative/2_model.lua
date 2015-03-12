----------------------------------------------------------------------
-- This script is used to build a convolutional neural nets to train
-- on surrogate classes.
--
-- 
-- 
--
-- By Group BearCat
----------------------------------------------------------------------
require 'torch'
require 'cunn'
require 'image'
require 'xlua'

-- number of surrogate classes
noutputs = 4000

-- input dimensions
nfeats = 3
width = 32
height = 32
ninputs = nfeats*width*height

-- hidden units, filter sizes
nstates = {64,128,256,512}
filtsize = 5
poolsize = 2

print '==> construct model'
model = nn.Sequential()

-- stage 1 : filter bank -> ReLU -> Max pooling
model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

-- stage 2 : filter bank -> ReLU -> Max pooling
model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

-- stage 3 : filter bank -> ReLU
model:add(nn.SpatialConvolutionMM(nstates[2], nstates[3], filtsize, filtsize))
model:add(nn.ReLU())

-- stage 4 : standard 2-layer neural network with dropout
model:add(nn.View(nstates[3]))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(nstates[3], nstates[4]))
model:add(nn.ReLU())
model:add(nn.Linear(nstates[4], noutputs))

-- define loss
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()	

