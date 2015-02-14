-- load required packages
require 'torch'
require 'image'
require 'xlua'
matio = require 'matio'

-- define functions
function preprocess(X, y)
   -- convert to greyscale 
   local X = X:float()
   local n = (#X)[1]
   data = {
      X = torch.Tensor(n, 1, 96, 96),
      y = y:reshape(n),
      size = n
   }
   
   for i = 1, n do
      data.X[i] = image.rgb2y(X[i]:reshape(3, 96, 96)):transpose(2, 3)
   end

   return data
end

function patchify(data, n_patches, kW, kH, dW, dH, seed)
   -- extract patches from images
   -- basic var
   seed = seed or 73
   dW = dW or 1
   dH = dH or 1
   local h = data.X:size()[3]
   local w = data.X:size()[4]
   local n = data.size
   patches = torch.Tensor(n*n_patches, kW*kH)

   -- calculate total number of patches 
   local n_row = (h - kH) / dH + 1
   local n_col = (w - kW) / dW + 1
   local all_idx = torch.range(1, n_row * n_col)

   -- extract n_patches randomly from each image   
   torch.manualSeed(seed)
   for i = 1, n do   -- for each image
      xlua.progress(i, n)   -- display progress
      local first = (i-1)*n_patches+1   -- index of first patch for this image
      --print(first)
      local idx = torch.multinomial(all_idx, n_patches)   -- randomly choose n_patches
      for j = 1, n_patches do
	 local k = idx[j]
	 --print(k)
	 -- find position of the patch
	 local x = ((k-1) % n_col) * dW + 1
	 --print(x)
	 local y = torch.floor((k-1) / n_col) * dH + 1
	 --print(y)
	 -- copy patch to new tensor
	 patches[{{first+j-1}, {}}] = torch.Tensor(kW*kH):copy(data.X[{{i}, {}, {x, x+kW-1}, {y, y+kH-1}}])
      end
   end
   return patches
end

print '==> loading images'
train_file = '../train.mat'
--test_file = '../test.mat'
loaded = matio.load(train_file)

print '==> converting to greyscale'
trdata = preprocess(loaded.X, loaded.y)

print '==> extracting patches'
pats = patchify(trdata, 10, 16, 16)
