----------------------------------------------------------------------
-- This script is used to load STL-10 data and preprocess it to
-- facilitate training.
--
-- Preprocessing procedure include conversion to grayscale image,
-- augmenting images by rotation, scaling and flipping, and a patch
-- extraction process described in Miclut et al. 2014.
--
-- By Group BearCat
----------------------------------------------------------------------
require 'torch'
require 'nn'
require 'image'
require 'xlua'

data_dir = '/scratch/yx887/courses/ds-ga-1008/dl-a2/'
print('==> loading data')
trdata = torch.load(paths.concat(data_dir, 'train.t7'))
centroids = torch.load(paths.concat(data_dir, 'centroids.t7'))
n = trdata.X:size()[1]
k = centroids:size()[1]
normkernel = image.gaussian(5)

-- build first layer network
mod = nn.Sequential()
mod:add(nn.SpatialConvolution(1, k, 16, 16, 1, 1))
mod:add(nn.ReLU())
mod:add(nn.SpatialContrastiveNormalization(k, normkernel))
mod:add(nn.SpatialAveragePooling(9, 9, 9, 9))

-- set weights to centroids
conv = mod:get(1)
conv:reset(0)
conv.weight = centroids:reshape(k, 1, 16, 16)

print('==> feeding forward first layer')
feats = mod:forward(trdata.X[1])
