----------------------------------------------------------------------
-- This script 
-- By Group BearCat
----------------------------------------------------------------------

require 'torch'
require 'cunn'
require 'xlua'
require 'optim'

-- train on svm 
print('==> loading data')
trfile = '_tr_4500x4608.t7'
trdata = torch.load(trfile)
vafile = '_va_500x4608.t7'
vadata = torch.load(vafile)
trsize = trdata.size
vasize = vadata.size

print('==> training on nn')
n_inputs = trdata.data:size(2)
n_outputs = 10
model = nn.Sequential()
model:add(nn.Linear(n_inputs, n_outputs))
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()
model:cuda()
criterion:cuda()

--svm = nn.Linear(n_inputs, n_outputs)
--loss = nn.MultiMarginCriterion()
--svm:cuda()
--loss:cuda()

parameters,gradParameters = model:getParameters()
optimState = {
   learningRate = 0.003,
   weightDecay = 0,
   momentum = 0.9,
   learningRateDecay = 1e-6
}
optimMethod = optim.sgd

confusion = optim.ConfusionMatrix({1, 2, 3, 4, 5, 6, 7, 8, 9, 10})

function train()
   model:training()
   epoch = epoch or 1
   shuffle = torch.randperm(trsize)
   for t = 1, trsize do
      xlua.progress(t, trsize)
      local input = trdata.data[shuffle[t]]
      local target = trdata.labels[shuffle[t]][1]
      local feval = function(x)
         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()

         -- f is the average of all criterions
         local output = model:forward(input)
         local f = criterion:forward(output, target)
         model:backward(input, criterion:backward(output, target))
      
         confusion:add(output, target)
         -- return f and df/dX
         return f,gradParameters
      end

      optimMethod(feval, parameters, optimState)
   end
   --confusion:updateValids()
   --print('==> training accuracy = ' .. confusion.totalValid)
   print(confusion)
   confusion:zero()
   epoch = epoch + 1
end

function test()
   model:evaluate()
   print('==> testing on validation set:')
   for t = 1, vasize do
      -- disp progress
      xlua.progress(t, vasize)

      -- get new sample
      local input = vadata.data[t]
      local target = vadata.labels[t][1]

      -- test sample
      local pred = model:forward(input)
      confusion:add(pred, target)
   end
   print(confusion)
   confusion:zero()
end

print('==> training')
for i = 1, 2000 do
   train()
   test()
end