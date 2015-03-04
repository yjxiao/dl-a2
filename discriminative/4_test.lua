----------------------------------------------------------------------
-- This script is used to test our convolutional neural nets trained
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

print '==> defining test procedure'

-- test function
function test()
   -- local vars
   local time = sys.clock()

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on test set:')
   for t = 1,testData:size() do
      -- disp progress
      xlua.progress(t, testData:size())

      -- get new sample
      local input = testData.data[t]
      if opt.type == 'double' then input = input:double()
      elseif opt.type == 'cuda' then input = input:cuda() end
      local target = testData.labels[t]

      -- test sample
      local pred = model:forward(input)
      -- print("\n" .. target .. "\n")
      confusion:add(pred, target)
   end

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- update confusion matrix
   confusion:updateValids()
   print('==> testing accuracy = ' .. confusion.totalValid)

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}

   test_acc = confusion.totalValid   
   last_acc = last_acc or 0
   count = count or 0
   if test_acc <= last_acc then
      count = count + 1
   else
      count = 0
      last_acc = test_acc
   end
   if count > 10 then
      optimState.learningRate = optimState.learningRate / 3
      last_acc = test_acc
      count = 0
   end

   -- save best model   
   if test_acc > max_test_acc then
      max_test_acc = test_acc
      local filename = paths.concat(opt.save, 'model_opt.net')
      torch.save(filename, model)
   end

   -- next iteration:
   confusion:zero()
end