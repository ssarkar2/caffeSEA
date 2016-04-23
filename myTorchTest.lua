require 'torch'
require 'mattorch'
require 'hdf5'
require 'nn'
require 'cutorch'
require 'math'
require 'cunn'



print("loading model")
filename = '/scratch0/sem4/cmsc702/deepSEA/deepSEA_orig/DeepSEA/results_3000000/model/bestmodel.net'
model = torch.load(filename)
model:cuda()


print("loading data")
test_file = 'test.mat'
--loaded = hdf5.open(test_file,'r')
--testData = {
--   data = loaded:read('/testxdata'):all(),
--}

loaded = mattorch.load(test_file)
testData = {
    data = loaded['testxdata']:transpose(3,1),
    labels = loaded['testdata']:transpose(2,1),
    size = function() return tr_size end
}

predSize = testData.data:size(1)
noutputs = 919
batchSize = 1024
nfeats = 4
width = 1000
height = 1


print("making predictions")
torch.setdefaulttensortype('torch.FloatTensor')
alloutputs = torch.DoubleTensor(predSize, noutputs)

for t = 1,predSize,batchSize do
  print(t)
  collectgarbage()
  local inputs = torch.FloatTensor(math.min(batchSize, predSize-t+1), nfeats, width, height)


  -- create mini batch
  k = 1
  for i = t,math.min(t+batchSize-1,predSize) do
     -- load new sample
     input = testData.data[i]:float()
     inputs[k]= input
     k = k + 1
  end

  inputs = inputs:cuda()


  local output = model:forward(inputs):double()
  k=1
  for ii = t,math.min(t+batchSize-1,predSize) do
      alloutputs[ii] = output[k]
      k=k+1
  end
end


--Save predictions
f=hdf5.open( 'testmat' .. '.pred' .. '.h5', 'w')
f:write('/pred',alloutputs)
f:write('/gt',testData['labels'])
f:close()
