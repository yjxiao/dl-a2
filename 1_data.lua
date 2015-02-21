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

-- load required packages
require 'torch'
require 'image'
require 'xlua'
matio = require 'matio'

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('STL-10 Dataset Preprocessing')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-noaug', true, 'do not include augmented data')
   cmd:text()
   opt = cmd:parse(arg or {})
end

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

function augment(data)
   -- rotate, scale, flip images to do augmentation
   local n = data.size * 4
   augmented = {
      X = torch.Tensor(n, 1, 96, 96),
      y = torch.Tensor(n),
      size = n
   }

   for i = 1, data.size do
      xlua.progress(i, data.size)   -- display progress      
      local x = data.X[i]
      local label = data.y[i]
      -- rotate +/- 10
      augmented.X[(i-1)*4+1] = image.rotate(x, math.pi/18)
      augmented.X[(i-1)*4+2] = image.rotate(x, -math.pi/18)
      -- flip left/right
      augmented.X[(i-1)*4+3] = image.hflip(x)
      -- scale by 1/3
      local temp = image.scale(x, 128)
      augmented.X[(i-1)*4+4] = image.crop(temp, 16, 16, 112, 112)
      for j = 1, 4 do
	 augmented.y[(i-1)*4+j] = data.y[i]
      end
   end

   return augmented
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

-- load matlab file if t7 file is not present
if not paths.filep('train.t7') then
   print '==> loading images'
   train_file = '../train.mat'
   --test_file = '../test.mat'
   loaded = matio.load(train_file)
   print '==> converting to greyscale'
   trdata = preprocess(loaded.X, loaded.y)
   torch.save('train.t7', trdata)   
-- otherwise load t7 file directly
else
   print '==> loading data'   
   trdata = torch.load('train.t7')
end

if opt.noaug then
   -- augmenting image if augmented.t7 is not present
   if not paths.filep('augmented.t7') then
      print '==> augmenting images'
      augmented = augment(trdata)
      torch.save('augmented.t7', augmented)   
   else
      print '==> loading augmented data'   
      augmented = torch.load('augmented.t7')
   end
end

print '==> extracting patches'
n_p = 10
patch_size = 16
pats = patchify(trdata, n_p, patch_size, patch_size)
if opt.noaug then
   pats_aug = patchify(augmented, n_p, patch_size, patch_size)
   pats = torch.cat(pats, pats_aug, 1)   
end
