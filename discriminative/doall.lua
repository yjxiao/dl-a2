print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('STL-10 Surrogate Class Training/Testing')
cmd:text()
cmd:text('Options:')
-- global:
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 2, 'number of threads')
-- training:
cmd:option('-save', 'result', 'subdirectory to save/log experiments in')
cmd:option('-learningRate', 3e-4, 'learning rate at t=0')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0.9, 'momentum (SGD only)')
cmd:option('-type', 'cuda', 'type: double | float | cuda')
cmd:text()
opt = cmd:parse(arg or {})

-- nb of threads and fixed seed (for repeatable experiments)
if opt.type == 'float' then
   print('==> switching to floats')
   torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
end
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

---------------------------------------------------------------------
print('==> loading training data')

data_dir = '/scratch/yx887/courses/ds-ga-1008/dl-a2/'
src_dir = '/home/yx887/documents/ds-ga-1008/dl-a2/discriminative'

loaded = torch.load(paths.concat(data_dir, 'train2_4000x120.t7'))
n_im = 4000
trsize = 120 * n_im
y = torch.Tensor(trsize)
for i= 1, n_im do
   y[{{(i-1)*120+1, i*120}}] = i
end

trainData = {
   data = loaded[{{1, trsize}}]:reshape(trsize, 3, 32, 32),
   labels = y,
   size = function() return trsize end
}

print('==> loading test data')
loaded = torch.load(paths.concat(data_dir, 'test2_4000x30.t7'))
tesize = 30 * n_im
y = torch.Tensor(tesize)
for i= 1, n_im do
   y[{{(i-1)*30+1, i*30}}] = i
end
testData = {
   data = loaded[{{1,tesize}}]:reshape(tesize, 3, 32, 32),
   labels = y,
   size = function() return tesize end
}

----------------------------------------------------------------------
print '==> preprocessing data: subtract pixel wise mean'

mean = trainData.data:mean(1)
trainData.data:add(-1, mean:expand(trsize, 3, 32, 32))
testData.data:add(-1, mean:expand(tesize, 3, 32, 32))

torch.save('mean2_4000.t7', mean)
----------------------------------------------------------------------
print('==> executing all')

dofile(paths.concat(src_dir, '2_model.lua'))
dofile(paths.concat(src_dir, '3_train.lua'))
dofile(paths.concat(src_dir, '4_test.lua'))

----------------------------------------------------------------------

print('==> training!')
max_test_acc = 0
while true do
   train()
   test()
end