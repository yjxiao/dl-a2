----------------------------------------------------------------------
-- This script is used to build a convolutional neural nets to train
-- on surrogate classes.
--
-- 
-- 
-- 
--
-- By Group BearCat
----------------------------------------------------------------------

require 'torch'
require 'xlua'
require 'optim'

-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('STL-10 Surrogate Class Training/Optimization')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-save', 'result', 'subdirectory to save/log experiments in')
   cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
   cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
   cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
   cmd:option('-momentum', 0.9, 'momentum (SGD only)')
   cmd:text()
   opt = cmd:parse(arg or {})
end

if opt.type == 'cuda' then
   model:cuda()
   criterion:cuda()
end

-- Log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

if model then
   parameters,gradParameters = model:getParameters()
end

optimState = {
   learningRate = opt.learningRate,
   weightDecay = opt.weightDecay,
   momentum = opt.momentum,
   learningRateDecay = 1e-7
}
optimMethod = optim.sgd

classes = {}
for i = 1, 4000 do
   classes[i] = i
end
confusion = optim.ConfusionMatrix(classes)

print('==> defining training procedure')
function train()
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- shuffle at each epoch
   shuffle = torch.randperm(trsize)

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

   for t = 1, trsize, opt.batchSize do
      -- disp progress
      xlua.progress(t, trsize)
      
      -- create mini batch
      local batch_size = math.min(trsize-t+1, opt.batchSize)
      local inputs = torch.CudaTensor(batch_size, 3, 32, 32)
      local targets = torch.CudaTensor(batch_size)
      for i = t, t+batch_size-1 do
         -- load new sample
         inputs[i-t+1] = trainData.data[shuffle[i]]
         targets[i-t+1] = trainData.labels[shuffle[i]]
      end
      
      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- reset gradients
                       gradParameters:zero()

                       -- f is the average of all criterions
		       local outputs = model:forward(inputs)
		       local f = criterion:forward(outputs, targets)
		       model:backward(inputs, criterion:backward(outputs, targets))
		       confusion:batchAdd(outputs, targets)

                       -- normalize gradients and f(X)
                       -- f = f / inputs:size(1)

                       -- return f and df/dX
                       return f, gradParameters
                    end
      optimMethod(feval, parameters, optimState)
   end

   -- time taken
   time = sys.clock() - time
   time = time / trsize
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
   
   -- update confusion matrix
   confusion:updateValids()   
   print('==> training accuracy = ' .. confusion.totalValid)

   -- update logger
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}

   -- save/log current net
   local filename = paths.concat(opt.save, 'model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)

   -- next epoch
   confusion:zero()
   epoch = epoch + 1
end
