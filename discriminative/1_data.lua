----------------------------------------------------------------------
-- This script is used to load STL-10 unlabeled data and augmenting them
-- to train on surrogate labels.
--
-- Processing procedure includes extracting 32x32 patches,
-- augmenting patches by translation, rotation, scaling, and color
-- manipulation. Procedures are described in Dosovitskiy et al. 2014.
--
-- By Group BearCat
----------------------------------------------------------------------

-- load required packages
require 'torch'
require 'image'
require 'xlua'
require 'unsup'

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('STL-10 Dataset Processing')
   cmd:text()
   cmd:text('Options:')
   cmd:text()
   opt = cmd:parse(arg or {})
end

-- define functions
function translate(img, x, y, pat_size)
   local img_size = (#img)[2]
   local dist = math.floor((torch.rand(1)[1] * 0.4 - 0.2) * pat_size)
   if torch.rand(1)[1] > 0.5 then
      -- make sure do not cross boundary
      new_x = math.max(1, x+dist)
      new_x = math.min(img_size-pat_size+1, new_x)
      new_y = y
   else
      new_y = math.max(1, y+dist)
      new_y = math.min(img_size-pat_size+1, new_y)
      new_x = x
   end
   return new_x, new_y
end

function scale(img, x, y, pat_size)
   local img_size = (#img)[2]
   local factor = torch.rand(1)[1] * 0.7 + 0.7
   img_size = math.floor(img_size * factor)
   new_x = math.max(math.floor(x * factor), 1)
   new_y = math.max(math.floor(y * factor), 1)
   img = image.scale(img, img_size, img_size)
   -- make sure not cross boundaries
   new_x = math.min(img_size-pat_size+1, new_x)
   new_y = math.min(img_size-pat_size+1, new_y)
   return img, new_x, new_y
end

function rotate(img, x, y, pat_size)
   local img_size = (#img)[2]
   -- random angle
   local rad = torch.rand(1)[1] * math.pi * 2 / 9 - math.pi / 9
   img = image.rotate(img, rad)
   -- calculate rotated position
   local vx = x + pat_size / 2 - img_size / 2
   local vy = y + pat_size / 2 - img_size / 2
   local new_vx = vx*math.cos(rad) - vy*math.sin(rad)
   local new_vy = vx*math.sin(rad) + vy*math.cos(rad)
   -- make sure not cross boundaries
   new_x = math.max(math.floor(new_vx + img_size / 2 - pat_size / 2), 1)
   new_y = math.max(math.floor(new_vy + img_size / 2 - pat_size / 2), 1)
   new_x = math.min(img_size-pat_size+1, new_x)
   new_y = math.min(img_size-pat_size+1, new_y)
   return img, new_x, new_y
end

function contrast1(pat, pat_size)
   local factors = torch.rand(1, 3):mul(1.5):add(0.5)
   local unfolded = pat:reshape(3, pat_size*pat_size):transpose(1, 2)
   ce, cv = unsup.pcacov(unfolded)
   local proj = unfolded * cv
   proj:cmul(torch.expand(factors, pat_size*pat_size, 3))
   pat = (proj * torch.inverse(cv)):transpose(1, 2):reshape(3, pat_size, pat_size)
   return pat
end

function contrast2(pat)
   local powfac = torch.rand(1)[1] * 3.75 + 0.25
   local mulfac = torch.rand(1)[1] * 0.7 + 0.7
   local addval = torch.rand(1)[1] * 0.2 - 0.1
   pat = image.rgb2hsv(pat)
   pat[{{2,3}, {}, {}}]:pow(powfac):mul(mulfac):add(addval)
   for i = 2, 3 do
      local max = pat[i]:max()
      local min = pat[i]:min()
      if min < 0 then
         pat[i]:add(-min)
      end
      if max > 1 then
         pat[i]:div(max)
      end
   end
   return pat
end

function color_mod(pat)
   local addval = torch.rand(1)[1] * 0.2 - 0.1
   pat[1]:add(addval)
   local min = pat[1]:min()
   local max = pat[1]:max()
   if min < 0 then
      pat[1]:add(-min)
   elseif max > 1 then
      pat[1]:div(max)
   end
   pat = image.hsv2rgb(pat)
   return pat
end

function random_aug(img, x, y, pat_size, k)
   local k = k or 9
   augmented = torch.Tensor(k, 3, pat_size, pat_size)
   augmented[1] = img[{{}, {x, x+pat_size-1}, {y, y+pat_size-1}}]
   for i = 2, k do
      local aux = img:clone()
      new_x, new_y = translate(aux, x, y, pat_size)
      aux, new_x, new_y = scale(aux, new_x, new_y, pat_size)
      aux, new_x, new_y = rotate(aux, new_x, new_y, pat_size)
      local pat = aux[{{}, {new_x, new_x+pat_size-1}, {new_y, new_y+pat_size-1}}]
      pat = contrast1(pat, pat_size)
      pat = contrast2(pat)
      pat = color_mod(pat)
      augmented[i] = pat
   end   
   return augmented
end

function patchify(data, patch_size, k)
   -- extract patches from images
   -- basic var
   local k = k or 9   -- number of different augmentations
   local h = data:size()[3]
   local w = data:size()[4]
   local n = data:size()[1]
   local n_c = data:size()[2]
   local patch_size = patch_size or 48
   patches = torch.Tensor(n*k, n_c, patch_size, patch_size)
   local kernel = torch.Tensor({-1, 0, 1}):reshape(3, 1)   -- to calculate gradient

   -- extract patches randomly from each image   
   for i = 1, n do   -- for each image
      -- xlua.progress(i, n)   -- display progress
      local first = (i-1)*k+1   -- index of first patch for this image
      -- convolve image to compute gradient
      local grad = image.convolve(data[i], kernel):abs()
      local avg_grad = grad:mean()
      -- "considerable" gradient
      local count = 0
      local alpha = 2
      repeat
         x = math.random(9, h-10-patch_size)
      	 y = math.random(9, w-8-patch_size)
	 count = count + 1
	 if count % 20 == 0 then
	    alpha = alpha / 1.1
	 end
      until grad[{{}, {x, x+patch_size-1}, {y, y+patch_size-1}}]:mean() > avg_grad * alpha
      patches[{{first, first+k-1}, {}, {}, {}}] = random_aug(data[i], x, y, patch_size, k)
   end
   return patches
end

print('==> loading images')
load_dir = '/scratch/courses/DSGA1008/A2/binary'
unlabeled_file = 'unlabeled_X.bin'
save_dir = '/scratch/yx887/courses/ds-ga-1008/dl-a2'

-- Open the files and set little endian encoding
data_fd = torch.DiskFile(paths.concat(load_dir, unlabeled_file), "r", true)
data_fd:binary():littleEndianEncoding()

-- Create and read the data
data = torch.ByteTensor(100000, 3, 96, 96)
data_fd:readByte(data:storage())
print('==> loaded')
-- Because data is in column-major, transposing the last 2 dimensions gives result that can be correctly visualized

n_unlabeled = data:size()[1]
patch_size = 32
n_used = 10000
batch_size = 500
n_batch = 20
n_trans = 150
idx = torch.randperm(n_unlabeled)[{{1, n_used}}]
data_aux = torch.Tensor(n_used, 3, 96, 96)
print('==> copying selected data')
for i = 1, n_used do
   xlua.progress(i, n_used)
   data_aux[i] = data[idx[i]]:double():transpose(2, 3)
end
print('==> augmenting data')
for i = 1, n_batch do
   xlua.progress(i, n_batch)
   file_name = 'augmented_unfolded_'..i..'.t7'
   torch.save(paths.concat(save_dir, file_name), patchify(data_aux[{{(i-1)*batch_size+1, i*batch_size}}], patch_size, n_trans):reshape(75000,3*32*32))
end

