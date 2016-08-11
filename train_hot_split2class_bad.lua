require 'xlua'
require 'optim'
require 'nn'
require 'image'

local matio = require 'matio'
dofile './provider.lua'
local c = require 'trepl.colorize'

opt = lapp[[
   -s,--save                  (default "logs_hot")      subdirectory to save logs
   -b,--batchSize             (default 128)          batch size
   -r,--learningRate          (default 1)        learning rate
   --learningRateDecay        (default 1.0e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.5)         momentum
   --epoch_step               (default 25)          epoch step
   --model                    (default vgg_split2class_bad)     model name
   --max_epoch                (default 5000)           maximum number of iterations
   --backend                  (default nn)            backend
   --type                     (default cuda)          cuda/float/cl
]]

print(opt)


nRepeat = 1
labelLength = 5
label0 = torch.cat({torch.ones(nRepeat), torch.zeros(nRepeat), torch.zeros(nRepeat), torch.zeros(nRepeat), torch.ones(nRepeat)}, 1)
label1 = torch.cat({torch.zeros(nRepeat), torch.zeros(nRepeat), torch.ones(nRepeat), torch.zeros(nRepeat), torch.ones(nRepeat)}, 1)
train_Y = torch.cat(torch.repeatTensor(label0, 5000, 1), torch.repeatTensor(label1, 5000, 1), 1)
test_Y = torch.cat(torch.repeatTensor(label0, 500, 1), torch.repeatTensor(label1, 500, 1), 1)  


do -- data augmentation module
  local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

  function BatchFlip:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchFlip:updateOutput(input)
    if self.train then
      local bs = input:size(1)
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end
      end
    end
    self.output:set(input)
    return self.output
  end
end

local function cast(t)
   if opt.type == 'cuda' then
      require 'cunn'
      return t:cuda()
   elseif opt.type == 'float' then
      return t:float()
   elseif opt.type == 'cl' then
      require 'clnn'
      return t:cl()
   else
      error('Unknown type '..opt.type)
   end
end

print(c.blue '==>' ..' configuring model')
local model = nn.Sequential()
model:add(nn.BatchFlip():float())
model:add(cast(nn.Copy('torch.FloatTensor', torch.type(cast(torch.Tensor())))))
model:add(cast(dofile('models/'..opt.model..'.lua')))
model:get(2).updateGradInput = function(input) return end

if opt.backend == 'cudnn' then
   require 'cudnn'
   cudnn.convert(model:get(3), cudnn)
end

print(model)

print(c.blue '==>' ..' loading data')
provider = torch.load 'provider.t7'
provider.trainData.data = provider.trainData.data:float()
provider.trainData.labels = train_Y
provider.testData.data = provider.testData.data:float()
provider.testData.labels = test_Y
print(#provider.trainData.labels)

--confusion = optim.ConfusionMatrix(10)

print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
testLogger.showPlot = false

parameters,gradParameters = model:getParameters()


print(c.blue'==>' ..' setting criterion')
criterion = cast(nn.SpatialClassNLLCriterion())


print(c.blue'==>' ..' configuring optimizer')
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}


function train()
  model:training()
  epoch = epoch or 1

  -- drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  --local targets = cast(torch.FloatTensor(opt.batchSize))
  local indices = torch.randperm(provider.trainData.data:size(1)):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil

  local tic = torch.tic()
  error = 0
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)
    
    local inputs = provider.trainData.data:index(1,v)
    --inputs = inputs:cuda()
    --print(#provider.trainData.labels:index(1,v))
    --targets:copy(provider.trainData.labels:index(1,v))
    local targets = cast(provider.trainData.labels:index(1,v))


    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      
      local outputs = model:forward(inputs)
      local f = criterion:forward(outputs, targets)
      error = error + f
      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)

      --confusion:batchAdd(outputs, targets)

      return f,gradParameters
    end
    optim.sgd(feval, parameters, optimState)
  end
  print('Current Error: '.. error/provider.trainData.data:size(1))

  --confusion:updateValids()
  --print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        --confusion.totalValid * 100, torch.toc(tic)))

  --train_acc = confusion.totalValid * 100

  --confusion:zero()
  epoch = epoch + 1
end

function accuracy(outputs)
  model:evaluate()
  --print(outputs[1])
  --print(label0)
  label0 = label0:cuda()
  label1 = label1:cuda()
  --print((outputs[1] - label0))
  correct = 0
print(outputs[1])
  for i=1,500 do
    local prediction = outputs[i]
    if (prediction - label0)*(prediction - label0)< (prediction - label1) * (prediction -label1) then
      correct = correct + 1
    end
  end

  for i=501,1000 do
    local prediction = outputs[i]
    if (prediction - label1)*(prediction - label1) < (prediction - label0)*(prediction - label0) then
      correct = correct + 1
    end
  end
  return correct/10
end

require "csvigo"
function test()
  -- disable flips, dropouts and batch normalization
  local layer_XA = model.modules[3].modules[4].modules[1].modules[4]
  local layer_XB = model.modules[3].modules[4].modules[2].modules[4]
  local layer_common = model.modules[3].modules[4].modules[3].modules[4]

  model:evaluate()
  print(c.blue '==>'.." testing")
  local bs = 125
  local outputSum = torch.Tensor(1000,labelLength):cuda()
  for i=1,provider.testData.data:size(1),bs do
    local outputs = model:forward(provider.testData.data:narrow(1,i,bs))
	if epoch == 0 then
		if outs_XA == nil then 
			outs_XA = layer_XA.output:clone()
			outs_XB = layer_XB.output:clone()
			outs_common = layer_common.output:clone() 
		else
			outs_XA = torch.cat(outs_XA, layer_XA.output:clone(), 1)
			outs_XB = torch.cat(outs_XB, layer_XB.output:clone(), 1)
			outs_common = torch.cat(outs_common, layer_common.output:clone(), 1)
		end
	end
    outputSum[{{i, i+bs - 1},{}}] = outputs
    --outputSum = torch.cat(outputSum, outputs,1)
    --confusion:batchAdd(outputs, provider.testData.labels:narrow(1,i,bs))
  end
  if outs_XA ~= nil then
	--convert tensor to table in order to save to csv 
	--print(weights_.weight)
	local t2 = {}
	local t3 = {}
	local t4 = {}
	for i=1,outs_XA:size(1) do
	  t2[i] = {}
	  t3[i] = {}
	  t4[i] = {}
	  for j=1,outs_XA:size(2) do
		t2[i][j] = outs_XA[i][j]
	    t3[i][j] = outs_XB[i][j]
	    t4[i][j] = outs_common[i][j]
  		end	
	end
		csvigo.save{path="visualize_exclusiveA.csv", data=t2}
	    csvigo.save{path="visualize_exclusiveB.csv", data=t3} 
		csvigo.save{path="visualize_common.csv", data=t4}
		outs_XA = nil
		outs_XB = nil
		outs_common = nil 	
  end

 --confusion:updateValids()
  result = accuracy(outputSum)
  print('Test accuracy:', result)
  
  if testLogger then
    paths.mkdir(opt.save)
    testLogger:add{result}
    testLogger:style{'-'}
    testLogger:plot()

    if paths.filep(opt.save..'/test.log.eps') then
      local base64im
      do
        os.execute(('convert -density 200 %s/test.log.eps %s/test.png'):format(opt.save,opt.save))
        os.execute(('openssl base64 -in %s/test.png -out %s/test.base64'):format(opt.save,opt.save))
        local f = io.open(opt.save..'/test.base64')
        if f then base64im = f:read'*all' end
      end

      local file = io.open(opt.save..'/report.html','w')
      file:write(([[
      <!DOCTYPE html>
      <html>
      <body>
      <title>%s - %s</title>
      <img src="data:image/png;base64,%s">
      <h4>optimState:</h4>
      <table>
      ]]):format(opt.save,epoch,base64im))
      for k,v in pairs(optimState) do
        if torch.type(v) == 'number' then
          file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
        end
      end
      file:write'</table><pre>\n'
      --file:write(tostring(confusion)..'\n')
      file:write(tostring(model)..'\n')
      file:write'</pre></body></html>'
      file:close()
    end
  end

  -- save model every 50 epochs
  if epoch % 50 == 0 then
    local filename = paths.concat(opt.save, 'model.net')
    print('==> saving model to '..filename)
    torch.save(filename, model:get(3):clearState())
  end

  --confusion:zero()
end


for i=1,opt.max_epoch do
  train()
  test()
end


