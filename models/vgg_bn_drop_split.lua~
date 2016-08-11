require 'nn'
local MaxPooling = nn.SpatialMaxPooling

local main = nn.Sequential()

-- building block
local function ConvBNReLU2(nInputPlane, nOutputPlane)
  main:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  main:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  main:add(nn.Tanh(true))
  return main
end
ConvBNReLU2(3,64)
ConvBNReLU2(64,64)
main:add(MaxPooling(2,2,2,2):ceil())

local vgg = nn.Sequential()

-- building block
local function ConvBNReLU(nInputPlane, nOutputPlane)
  vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  vgg:add(nn.Tanh(true))
  return vgg
end
ConvBNReLU(64,64)
ConvBNReLU(64,64)
vgg:add(MaxPooling(2,2,2,2):ceil())

ConvBNReLU(64,64)
ConvBNReLU(64,64)
ConvBNReLU(64,64)
vgg:add(MaxPooling(2,2,2,2):ceil())

ConvBNReLU(64,64)
ConvBNReLU(64,64)
ConvBNReLU(64,64)
vgg:add(MaxPooling(2,2,2,2):ceil())

--[[main:add(vgg)
-- '#' prints the dimension of the matrix
print(#main:cuda():forward(torch.CudaTensor(100,3,32,32)))]]
--view unrolls 2d to 1d for fully connected neural network
--uncomment and run to see the correct dimension for nn.View. Remember to ignore the batch size. EG) 100*128*2 =>nn.View = 128*2
vgg:add(nn.View(128*2))

--this is Hapha, Hbeta, Hgamma
local vgg2 = vgg:clone()
local vgg3 = vgg:clone()
local concat = nn.Concat(2)

--this is the cully connected neural network right before the label. function f
classifier = nn.Sequential()
classifier:add(nn.Linear(128*2,64))
classifier:add(nn.BatchNormalization(64))
classifier:add(nn.Tanh(true))
classifier:add(nn.Linear(64,1))
classifier:add(nn.Sigmoid())

--here clone is not tied weight. its just a copy. Classifier3,4,5 are tied.
classifier3 = classifier:clone()
classifier4 = classifier3:clone('weight','bias','gradWeight','gradBias')
classifier5 = classifier3:clone('weight','bias','gradWeight','gradBias')
classifier2 = classifier:clone('weight','bias','gradWeight','gradBias')

--concatenate fully classifer and classifier 4 right behind vgg (paralle modules). output label follow concat 
local concatvgg = nn.Concat(2)
concatvgg:add(classifier)
concatvgg:add(classifier4)
vgg:add(concatvgg)

local concatvgg2 = nn.Concat(2)
concatvgg2:add(classifier2)
concatvgg2:add(classifier5)
vgg2:add(concatvgg2)

vgg3:add(classifier3)	

concat:add(vgg)
concat:add(vgg2)
concat:add(vgg3)
main:add(concat)


-- initialization from MSR
local function MSRinit(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      v.bias:zero()
    end
  end
  -- have to do for both backends
  init'nn.SpatialConvolution'
end

MSRinit(main)

-- check that we can propagate forward without errors
-- should get 16x10 tensor
print(#main:cuda():forward(torch.CudaTensor(100,3,32,32)))

return main:cuda()
