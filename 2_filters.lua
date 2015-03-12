----------------------------------------------------------------------
-- This script is used to process patches extracted from training data
-- and run k-means on these patches to acquire filters.
--
-- Processing procedure include patch-wise normalization and ZCA
-- whitening. K-means clustering on processed patches is performed
-- as described in Miclut et al. 2014.
--
-- By Group BearCat
----------------------------------------------------------------------

-- load required packages
require 'torch'
require 'unsup'
require 'xlua'

function normalize(patches)
   -- perform patch-wise normalization
   local n = (#patches)[1]
   local d = (#patches)[2]
   local max = patches:max(2)
   max[torch.eq(max, 0)] = 1
   local xmax = max:expand(n, d)
   patches:cdiv(xmax)
   local mean = patches:mean(2)
   local xmean = mean:expand(n, d)
   patches:add(-1, xmean)
end

-- normalize and zca whitening, then use k-means to find centroids
data_dir = '/scratch/yx887/courses/ds-ga-1008/dl-a2/'
pat_file = paths.concat(data_dir, 'patches.t7')
print('==> loading patches')
pats = torch.load(pat_file)
print('==> normalizing patches')
normalize(pats)
print('==> zca whitening')
whitened_pats = unsup.zca_whiten(pats)
k = 300
patchsize = 16
print('==> calculating centroids')
cents = unsup.kmeans(whitened_pats, k, 5000):reshape(k, patchsize, patchsize)
cent_file = paths.concat(data_dir, 'centroids.t7')
print('==> saving centroids')
torch.save(cent_file, cents)
