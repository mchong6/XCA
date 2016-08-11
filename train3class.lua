require 'xlua'
require 'optim'
require 'nn'
require 'image'
require 'gnuplot'

local matio = require 'matio'
dofile './provider2.lua'
local c = require 'trepl.colorize'

opt = lapp[[
   -s,--save                  (default "logs_three")      subdirectory to save logs
   -b,--batchSize             (default 128)          batch size
   -r,--learningRate          (default 1)        learning rate
   --learningRateDecay        (default 1.0e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 25)          epoch step
   --model                    (default vgg_bn_drop2_3class)     model name
   --max_epoch                (default 1000)           maximum number of iterations
   --backend                  (default nn)            backend
   --type                     (default cuda)          cuda/float/cl
]]

print(opt)

W_Decor_rate = 5
W_Decor_mult = 5	
nRepeat = 4
labelLength = 3 * nRepeat
label0 = torch.cat({torch.ones(nRepeat), torch.zeros(nRepeat), torch.zeros(nRepeat)}, 1)
label1 = torch.cat({torch.zeros(nRepeat), torch.ones(nRepeat), torch.zeros(nRepeat)}, 1)
label2 = torch.cat({torch.zeros(nRepeat), torch.zeros(nRepeat), torch.ones(nRepeat)}, 1)



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
model = nn.Sequential()
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
provider = torch.load 'provider2.t7'
provider.trainData.data = provider.trainData.data:float()
provider.testData.data = provider.testData.data:float()

--confusion = optim.ConfusionMatrix(10)

print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
testLogger.showPlot = false

parameters,gradParameters = model:getParameters()


print(c.blue'==>' ..' setting criterion')
criterion = cast(nn.MSECriterion())


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
  if epoch % opt.epoch_step == 0 then 
	optimState.learningRate = optimState.learningRate/2
	--W_Decor_rate = math.ceil(W_Decor_rate / 2)
	--W_Decor_mult = math.ceil(W_Decor_mult / 2)
  end
  
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  --local targets = cast(torch.FloatTensor(opt.batchSize))
  local indices = torch.randperm(provider.trainData.data:size(1)):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil

  local tic = torch.tic()
  error = 0
  error_decor = 0
  --switch = 1
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

	local lastlayer = model.modules[model:size(1)].modules[model.modules[model:size(1)]:size(1)].modules[model.modules[model:size(1)].modules[model.modules[model:size(1)]:size(1)]:size(1)-1]

	local outputs = model:forward(inputs)
	local f = criterion:forward(outputs, targets)
	local df_do, grad_Decor = criterion:backward(outputs, targets, switch)
	model:backward(inputs, df_do)
	lastlayer.gradWeight:add(grad_Decor * W_Decor_mult)

					
	return f ,gradParameters
  end
    optim.sgd(feval, parameters, optimState)
  end

  if MSEPLOT == nil then
      MSEPLOT = torch.Tensor(1):fill(error/provider.trainData.data:size(1)):cuda()
      WEIGHTPLOT = torch.Tensor(1):fill(error_decor/provider.trainData.data:size(1)):cuda()
  else
      MSEPLOT = torch.cat(MSEPLOT, torch.Tensor(1):fill(error/provider.trainData.data:size(1)):cuda(), 1)
      WEIGHTPLOT = torch.cat(WEIGHTPLOT, torch.Tensor(1):fill(error_decor/provider.trainData.data:size(1)):cuda(), 1)
  end
  print('Current Error: '.. error/provider.trainData.data:size(1))
  print('Decorrelated Error: '.. error_decor/provider.trainData.data:size(1))
  print('Toal Error: '.. (error_decor + error)/provider.trainData.data:size(1))
  epoch = epoch + 1
end

function accuracy(outputs)
  model:evaluate()
  label0 = label0:cuda()
  label1 = label1:cuda()
  label2 = label2:cuda()
  correct = 0
  for i=1,1000 do
    local prediction = outputs[i]
    if (prediction - label0)*(prediction - label0)< (prediction - label1) * (prediction -label1)  and (prediction - label0)*(prediction - label0) < (prediction - label2)*(prediction - label2) then
      correct = correct + 1
    end
  end

  for i=1001,2000 do
    local prediction = outputs[i]
    if (prediction - label1)*(prediction - label1) < (prediction - label0)*(prediction - label0) and (prediction - label1)*(prediction - label1) < (prediction - label2)*(prediction - label2)then
      correct = correct + 1
    end
  end
 
  for i= 2001,3000 do
    local prediction = outputs[i]
    if (prediction - label2)*(prediction - label2) < (prediction - label0)*(prediction - label0) and (prediction - label2)*(prediction - label2) < (prediction - label1)*(prediction - label1)then
      correct = correct + 1
    end
  end

  return correct / 3000 * 100
end


function test()
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  print(c.blue '==>'.." testing")
  local bs = 125
  local outputSum = torch.Tensor(3000,labelLength):cuda()
  for i=1,provider.testData.data:size(1),bs do
    local outputs = model:forward(provider.testData.data:narrow(1,i,bs))
    outputSum[{{i, i+bs - 1},{}}] = outputs
    --outputSum = torch.cat(outputSum, outputs,1)
    --confusion:batchAdd(outputs, provider.testData.labels:narrow(1,i,bs))
  end

 --confusion:updateValids()
  local result = accuracy(outputSum)
  print('Test accuracy:', result)

  gnuplot.epsfigure(opt.save.. '/testMSE.eps')
  gnuplot.plot('MSE',MSEPLOT, '-')
  gnuplot.plotflush()

  gnuplot.epsfigure(opt.save.. '/testWEIGHT.eps')
  gnuplot.plot('MSE',WEIGHTPLOT, '-')
  gnuplot.plotflush()

  
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

end


for i=1,opt.max_epoch do
  train()
  test()
end


