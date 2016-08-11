require 'nn'

local MaxPooling = nn.SpatialMaxPooling

local main = nn.Sequential()

-- building block
local function ConvBNReLU2(nInputPlane, nOutputPlane)
  main:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  main:add(nn.Tanh(true))
  return main
end
--ConvBNReLU2(3,3)
--main:add(MaxPooling(2,2,2,2):ceil())


local vgg = nn.Sequential()

-- building block
local function ConvBNReLU(nInputPlane, nOutputPlane)
  vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  vgg:add(nn.Tanh(true))
  return vgg
end
ConvBNReLU(3,3)
vgg:add(MaxPooling(2,2,2,2):ceil())
ConvBNReLU(3,3)
vgg:add(MaxPooling(2,2,2,2):ceil())
vgg:add(nn.View(3*8*8))

local vgg2 = vgg:clone()
local vgg3 = vgg:clone()
local vgg4 = vgg:clone()
local vgg5 = vgg:clone()
local concat = nn.Concat(2)

classifier = nn.Sequential()
classifier:add(nn.Linear(3*8*8,1))
classifier:add(nn.Sigmoid())

classifier3 = classifier:clone()
classifierAC = classifier:clone()
classifier4 = classifier3:clone('weight','bias','gradWeight','gradBias')
classifier5 = classifier3:clone('weight','bias','gradWeight','gradBias')
classifier7 = classifier3:clone('weight','bias','gradWeight','gradBias')
classifier8 = classifier3:clone('weight','bias','gradWeight','gradBias')
--classifier2 = classifier:clone('weight','bias','gradWeight','gradBias')
classifier2 = classifier:clone()
--classifier6 = classifier:clone('weight','bias','gradWeight','gradBias')
classifier6 = classifier:clone()
classifierABG = classifier:clone()
classifierACG = classifier:clone()

classifierAC2 = classifierAC:clone('weight','bias','gradWeight','gradBias')
classifierAC3 = classifierAC:clone('weight','bias','gradWeight','gradBias')
classifierAC4 = classifierAC:clone('weight','bias','gradWeight','gradBias')
classifierAC5 = classifierAC:clone('weight','bias','gradWeight','gradBias')

local concatvgg = nn.Concat(2)
concatvgg:add(classifier)
concatvgg:add(classifier4)
concatvgg:add(classifierAC)
vgg:add(concatvgg)

local concatvgg2 = nn.Concat(2)
concatvgg2:add(classifier2)
concatvgg2:add(classifier5)
concatvgg2:add(classifierAC2)
vgg2:add(concatvgg2)

local concatvgg3 = nn.Concat(2)
concatvgg3:add(classifier6)
concatvgg3:add(classifier7)
concatvgg3:add(classifierAC3)
vgg4:add(concatvgg3)

local concatvgg4 = nn.Concat(2)
concatvgg4:add(classifier3)
concatvgg4:add(classifierAC5) --connect to AC NN
concatvgg4:add(classifierABG)  --connect to exclusive NN
vgg3:add(concatvgg4)

local concatvgg5 = nn.Concat(2)
concatvgg5:add(classifierAC4)
concatvgg5:add(classifier8)   --connect to AB NN
concatvgg5:add(classifierACG)
vgg5:add(concatvgg5)

concat:add(vgg)
concat:add(vgg2)
concat:add(vgg4)
concat:add(vgg3)
concat:add(vgg5)
main:add(concat)

--print(#main:cuda():forward(torch.CudaTensor(128,3,32,32)))
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

MSRinit(vgg)

-- check that we can propagate forward without errors
-- should get 16x10 tensor
--print(#main:cuda():forward(torch.CudaTensor(100,3,32,32)))

return main:cuda()
