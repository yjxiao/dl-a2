----------------------------------------------------------------------
-- This script 
-- By Group BearCat
----------------------------------------------------------------------

require 'torch'
require 'cunn'
require 'xlua'
matio = require 'matio'

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('STL-10 Dataset Processing')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-type', 'cuda', 'type: double | float | cuda')
   cmd:text()
   opt = cmd:parse(arg or {})
end

-- load training data and separate to training/validation
print('==> loading training data')
data_dir = '/home/yx887/documents/ds-ga-1008/dl-a2/discriminative'
loaded = matio.load(paths.concat(data_dir, 'train.mat'))
n = loaded.X:size(1)
trsize = 4500
vasize = 500
loaded.X = loaded.X:double():div(255):reshape(n, 3, 96, 96):transpose(3, 4)

trdata = {
   data = loaded.X[{{1, trsize}}],
   labels = loaded.y[{{1, trsize}}],
   size = trsize
}
vadata = {
   data = loaded.X[{{trsize+1, trsize+vasize}}],
   labels = loaded.y[{{trsize+1, trsize+vasize}}],
   size = vasize
}

-------------------------------------------------------------------
print('==> loading model')
mod_dir = '/home/yx887/documents/ds-ga-1008/dl-a2/discriminative'
model_file = 'model_2000.net'
model = torch.load(paths.concat(mod_dir, model_file))
if opt.type == 'cuda' then
   model:cuda()
elseif opt.type == 'double' then
   model:double()
end
mean_file = 'mean_2000.t7'
mean = torch.load(paths.concat(mod_dir, mean_file))

function get_features(data, mean, model, stride)
   local n = data:size(1)   -- total number of images
   local im_size = data:size(3)   -- image size
   local patch_size = mean:size(3)   -- patch size (model input size)
   local stride = stride or 4
   local size = math.floor((im_size - patch_size) / stride) + 1   -- output feature map size before pooling
   local n_pools = 3

   feats = torch.CudaTensor(n, 512, n_pools, n_pools)   -- store features extracted by feeding to model
   local batch_size = 50
   local xmean = mean:expand(batch_size, 3, patch_size, patch_size)

   -- 4-quadrant max-pooling
   local pool_size = math.floor(size / n_pools)
   local pool = nn.SpatialMaxPooling(pool_size, pool_size, pool_size, pool_size)
   pool:cuda()

   for i = 1, n, batch_size do
      xlua.progress(i, n)
      local temp = torch.CudaTensor(batch_size, 512, size, size)	
      for x = 1, size do
         for y = 1, size do
	    -- current patch
            local patch = data[{{i, i+batch_size-1}, {}, {(x-1)*stride+1, (x-1)*stride+patch_size}, {(y-1)*stride+1, (y-1)*stride+patch_size}}]:clone():add(-1, xmean):cuda()
            -- get output from last hidden layer
            model:forward(patch)
            temp[{{}, {}, {x}, {y}}] = model:get(12).output:clone()
	 end
      end
      feats[{{i, i+batch_size-1}}] = pool:forward(temp)      
   end
   return feats:reshape(n, 512*n_pools*n_pools)
end

-- extract features using trained model
print('==> feeding data through network')
trdata.data = get_features(trdata.data, mean, model)
vadata.data = get_features(vadata.data, mean, model)

print('==> saving training and validation set')
torch.save('_tr_4500x4608.t7', trdata)
torch.save('_va_500x4608.t7', vadata)

