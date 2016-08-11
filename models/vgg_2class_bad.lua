require 'nn'

local MaxPooling = nn.SpatialMaxPooling

local main = nn.Sequential()

-- building block
local function ConvBNReLU2(nInputPlane, nOutputPlane)
  main:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  main:add(nn.ReLU(true))
  return main
end
ConvBNReLU2(3,3)
main:add(MaxPooling(2,2,2,2):ceil())


local vgg = nn.Sequential()

-- building block
local function ConvBNReLU(nInputPlane, nOutputPlane)
  vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  vgg:add(nn.Tanh(true))
  return vgg
end
ConvBNReLU(3,3)
vgg:add(MaxPooling(2,2,2,2):ceil())
vgg:add(nn.View(3*8*8))

classifier = nn.Sequential()
classifier:add(nn.Linear(3*8*8,2))
classifier:add(nn.Sigmoid())
vgg:add(classifier)
main:add(vgg)

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
