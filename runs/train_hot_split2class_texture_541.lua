require 'xlua'
require 'optim'
require 'nn'
require 'image'
require "csvigo"
require 'gnuplot'

--seed random
torch.manualSeed(541)

local matio = require 'matio'
dofile './provider_texture.lua'
local c = require 'trepl.colorize'

opt = lapp[[
   -s,--save                  (default "CSV_541")      subdirectory to save logs
   -b,--batchSize             (default 128)          batch size
   -r,--learningRate          (default 1e-1)        learning rate
   --learningRateDecay        (default 1.0e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.5)         momentum
   --epoch_step               (default 25)          epoch step
   --model                    (default vgg_bn_drop_split)     model name
   --max_epoch                (default 90)           maximum number of iterations
   --backend                  (default nn)            backend
   --type                     (default cuda)          cuda/float/cl
]]

print(opt)


nRepeat = 1
labelLength = 5
label0 = torch.cat({torch.ones(nRepeat), torch.zeros(nRepeat), torch.zeros(nRepeat), torch.zeros(nRepeat), torch.ones(nRepeat)}, 1)
label1 = torch.cat({torch.zeros(nRepeat), torch.zeros(nRepeat), torch.ones(nRepeat), torch.zeros(nRepeat), torch.ones(nRepeat)}, 1)
train_Y = torch.cat(torch.repeatTensor(label0, 4000, 1), torch.repeatTensor(label1, 4000, 1), 1)
--test_Y = torch.cat(torch.repeatTensor(label0, 500, 1), torch.repeatTensor(label1, 500, 1), 1)  


testlabel0 = torch.cat({torch.zeros(nRepeat), torch.zeros(nRepeat), torch.zeros(nRepeat), torch.zeros(nRepeat), torch.ones(nRepeat)}, 1)
testlabel1 = torch.cat({torch.ones(nRepeat), torch.zeros(nRepeat), torch.zeros(nRepeat), torch.zeros(nRepeat), torch.zeros(nRepeat)}, 1)
testlabel2 = torch.cat({torch.zeros(nRepeat), torch.zeros(nRepeat), torch.ones(nRepeat), torch.zeros(nRepeat), torch.zeros(nRepeat)}, 1)

test_Y = torch.cat({torch.repeatTensor(label0, 1000, 1),torch.repeatTensor(label1, 1000, 1),torch.repeatTensor(testlabel0, 100, 1), torch.repeatTensor(testlabel1, 100, 1), torch.repeatTensor(testlabel2, 100, 1)}, 1)  

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
model:add(cast(dofile('../models/'..opt.model..'.lua')))
model:get(2).updateGradInput = function(input) return end

if opt.backend == 'cudnn' then
   require 'cudnn'
   cudnn.convert(model:get(3), cudnn)
end

print(model)

print(c.blue '==>' ..' loading data')
provider = torch.load 'provider_texture.t7'

provider.testData.data = torch.cat({provider.trainData.data[{{4001,5000}}]:float(),provider.trainData.data[{{9001,10000}}]:float(),provider.testData.data:float()},1)
provider.testData.labels = test_Y

provider.trainData.data = torch.cat(provider.trainData.data[{{1,4000}}]:float(), provider.trainData.data[{{5001,9000}}]:float(),1)
provider.trainData.labels = train_Y

print(#provider.testData.data)

image.save("train.png", provider.trainData.data[1])
image.save("train2.png", provider.trainData.data[4001])
image.save("test.png", provider.testData.data[2001])
image.save("test2.png", provider.testData.data[2101])
image.save("test3.png", provider.testData.data[2201])


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
	--print(outputs[10])
      local f = criterion:forward(outputs, targets)
      error = error + f
      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)

      --confusion:batchAdd(outputs, targets)

      return f,gradParameters
    end
    optim.sgd(feval, parameters, optimState)
  end
  if MSEPLOT == nil then
      MSEPLOT = torch.Tensor(1):fill(error/provider.trainData.data:size(1)):cuda()
  else
      MSEPLOT = torch.cat(MSEPLOT, torch.Tensor(1):fill(error/provider.trainData.data:size(1)):cuda(), 1)
  end
  print('Current Error: '.. error/provider.trainData.data:size(1))
  epoch = epoch + 1
end

function distance(input)
	--cat in second dimension because they are 1D
	labels = torch.cat({label0, label1, testlabel0, testlabel1, testlabel2}, 2) 
	distances = torch.Tensor(5)
	for i = 1, 5 do
		distances[i] = torch.dist(input, labels[{{},i}]) 
	end	
	y, index = torch.sort(distances)
	return index[1] --return index of smallest distance
end

function accuracy(outputs)
	model:evaluate()

	correct_norm = 0
	correct_ANB = 0
	correct_XA = 0
	correct_XB = 0
	print(outputs[1])
	print(outputs[1001])
	print(outputs[2001])
	for i = 1, 1000 do
		local prediction = outputs[i]
		if torch.dist(prediction, label0) < torch.dist(prediction, label1) then
			correct_norm = correct_norm + 1
		end
	end

	for i = 1001, 2000 do
		local prediction = outputs[i]
		if torch.dist(prediction, label1) < torch.dist(prediction, label0) then
			correct_norm = correct_norm + 1
		end
	end
	
	for i=2001,2100 do
		local prediction = outputs[i]
		if distance(prediction) == 3 then
	 		 correct_ANB = correct_ANB + 1
		end
	end

	for i=2101,2200 do
		local prediction = outputs[i]
		if distance(prediction) == 4 then
	  		correct_XA = correct_XA + 1
		end
	end

	for i= 2201,2300 do
		local prediction = outputs[i]
		if distance(prediction) == 5 then
	  		correct_XB = correct_XB + 1
		end
	end
	return correct_norm / 20, correct_ANB, correct_XA, correct_XB
end


function test()
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  local layer_XA = model.modules[3].modules[8].modules[1].modules[28]
  local layer_XB = model.modules[3].modules[8].modules[2].modules[28]
  local layer_common = model.modules[3].modules[8].modules[3].modules[28]

  print(c.blue '==>'.." testing")
	--convert to cuda
	label0 = label0:cuda()
	label1 = label1:cuda()
	testlabel0 = testlabel0:cuda()
	testlabel1 = testlabel1:cuda()
	testlabel2 = testlabel2:cuda()
  local bs = 100
  local outputSum = torch.Tensor(2300,labelLength):cuda()
  for i=1,provider.testData.data:size(1),bs do
    local outputs = model:forward(provider.testData.data:narrow(1,i,bs))
	if epoch % 30 == 0 and i <= 2000 then
		if outs_XA == nil then 
			outs_XA = layer_XA.output:clone()
			outs_XB = layer_XB.output:clone()
			outs_common = layer_common.output:clone() 
		else
			outs_XA = torch.cat(outs_XA, layer_XA.output:clone(), 1)
			outs_XB = torch.cat(outs_XB, layer_XB.output:clone(), 1)
			outs_common = torch.cat(outs_common, layer_common.output:clone(), 1)
		end
	elseif epoch% 30 == 0 then
		if outs_XA_X == nil then 				--output from XA filter with exlusive inputs
			outs_XA_X = layer_XA.output:clone()
			outs_XB_X = layer_XB.output:clone()
			outs_common_X = layer_common.output:clone() 
		else
			outs_XA_X = torch.cat(outs_XA_X, layer_XA.output:clone(), 1)
			outs_XB_X = torch.cat(outs_XB_X, layer_XB.output:clone(), 1)
			outs_common_X = torch.cat(outs_common_X, layer_common.output:clone(), 1)
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
	local t_XA_X = {}
	local t_XB_X = {}
	local t_common_X = {}
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
	for i=1,outs_XA_X:size(1) do
		t_XA_X[i] = {}
		t_XB_X[i] = {}
		t_common_X[i] = {}
		for j=1,outs_XA_X:size(2) do
			t_XA_X[i][j] = outs_XA_X[i][j]
			t_XB_X[i][j] = outs_XB_X[i][j]
			t_common_X[i][j] = outs_common_X[i][j]
  		end	
	end
	csvigo.save{path=opt.save.."/visualize_exclusiveA.csv", data=t2}
    csvigo.save{path=opt.save.."/visualize_exclusiveB.csv", data=t3} 
	csvigo.save{path=opt.save.."/visualize_common.csv", data=t4}
	csvigo.save{path=opt.save.."/visualize_exclusiveA_X.csv", data=t_XA_X}
    csvigo.save{path=opt.save.."/visualize_exclusiveB_X.csv", data=t_XB_X} 
	csvigo.save{path=opt.save.."/visualize_common_X.csv", data=t_common_X}
	outs_XA = nil
	outs_XB = nil
	outs_common = nil 	
	outs_XA_X = nil
	outs_XB_X = nil
	outs_common_X = nil

	local output_common = outputSum[{{2001,2100}}]   --output labels when put in exclusive inputs
	local output_XA = outputSum[{{2101,2200}}]
	local output_XB = outputSum[{{2201,2300}}]
	local z = {}
	local z2 = {}
	local z3 = {}
	for i=1,output_XA:size(1) do
		z[i] = {}
		z2[i] = {}
		z3[i] = {}
		for j=1,output_XA:size(2) do
			z[i][j] = output_XA[i][j]
			z2[i][j] = output_XB[i][j]
			z3[i][j] = output_common[i][j]
  		end	
	end
	csvigo.save{path=opt.save.."/exclusiveA_outputs.csv", data=z}
    csvigo.save{path=opt.save.."/exclusiveB_outputs.csv", data=z2} 
	csvigo.save{path=opt.save.."/common_outputs.csv", data=z3}
	output_XA = nil
	output_XB = nil
	output_common = nil 	
  end
 --confusion:updateValids()
  result1, result2, result3, result4 = accuracy(outputSum)
  print('Test accuracy:', result1)
  print('Common ANB Test accuracy:', result2)
	print('Exclusive A Test accuracy:', result3)
	print('Exclusive B Test accuracy:', result4)

  gnuplot.epsfigure(opt.save.. '/testMSE.eps')
  gnuplot.plot('MSE',MSEPLOT, '-')
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

  --confusion:zero()
end

for i=1,opt.max_epoch do
  train()
  test()
end


