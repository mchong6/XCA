require 'nn'
require 'image'
require 'xlua'
local matio = require 'matio'

local Provider = torch.class 'Provider'

function Provider:__init(full)
  local trsize = 10000
  local tesize = 1000


  -- load dataset
  self.trainData = {
     data = torch.Tensor(10000, 3072),
     labels = torch.Tensor(10000),
     size = function() return trsize end
  }
  local subset = matio.load('CIFAR2_reshaped.mat')
  nRepeat = 4
  labelLength = nRepeat * 2
  label0 = torch.cat(torch.ones(nRepeat), torch.zeros(nRepeat),1)
  label1 = torch.cat(torch.zeros(nRepeat), torch.ones(nRepeat),1)


  train_Y = torch.cat(torch.repeatTensor(label0, 5000, 1), torch.repeatTensor(label1, 5000, 1), 1)
  test_Y = torch.cat(torch.repeatTensor(label0, 500, 1), torch.repeatTensor(label1, 500, 1), 1)  



  local trainData = self.trainData
  --print(#subset.train_X)
  trainData.data = subset.train_X:double()
  --print(#trainData.data)
  trainData.labels = train_Y

  self.testData = {
     data = subset.test_X:double(),
     labels =test_Y:double(),
     size = function() return tesize end
  }
  local testData = self.testData
  --testData.labels = testData.labels + 1


end

function Provider:normalize()
  ----------------------------------------------------------------------
  -- preprocess/normalize train/test sets
  --
  local trainData = self.trainData
  local testData = self.testData

  print '<trainer> preprocessing data (color space + normalization)'
  collectgarbage()

  -- preprocess trainSet
  local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
  for i = 1,trainData:size() do
     xlua.progress(i, trainData:size())
     -- rgb -> yuv
     local rgb = trainData.data[i]
     local yuv = image.rgb2yuv(rgb)
     print(#yuv)
     -- normalize y locally:
     yuv[1] = normalization(yuv[{{1}}])
     trainData.data[i] = yuv
  end
  -- normalize u globally:
  local mean_u = trainData.data:select(2,2):mean()
  local std_u = trainData.data:select(2,2):std()
  trainData.data:select(2,2):add(-mean_u)
  trainData.data:select(2,2):div(std_u)
  -- normalize v globally:
  local mean_v = trainData.data:select(2,3):mean()
  local std_v = trainData.data:select(2,3):std()
  trainData.data:select(2,3):add(-mean_v)
  trainData.data:select(2,3):div(std_v)

  trainData.mean_u = mean_u
  trainData.std_u = std_u
  trainData.mean_v = mean_v
  trainData.std_v = std_v

  -- preprocess testSet
  for i = 1,testData:size() do
    xlua.progress(i, testData:size())
     -- rgb -> yuv
     local rgb = testData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[{1}] = normalization(yuv[{{1}}])
     testData.data[i] = yuv
  end
  -- normalize u globally:
  testData.data:select(2,2):add(-mean_u)
  testData.data:select(2,2):div(std_u)
  -- normalize v globally:
  testData.data:select(2,3):add(-mean_v)
  testData.data:select(2,3):div(std_v)
end
