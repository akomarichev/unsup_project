require 'torch'
require 'nn'
require 'optim'
require 'image'
local cuda = pcall(require, 'cutorch')
local mnist = require 'mnist'
local AE = require 'model'

local Xtrain = mnist.traindataset()
local Xtest = mnist.testdataset()

testset = Xtest.data:double():div(255)

trainset = Xtrain.data[{{1, 50000}}]:double():div(255)

validationset = Xtrain.data[{{50001, 60000}}]:double():div(255)

local Ntrain = trainset:size(1)
local Ntest = testset:size(1)
local Nvalid = validationset:size(1)

AE:createAutoencoder(trainset)
local model = AE.autoencoder

criterion = nn.BCECriterion()

if cuda then
  trainset = trainset:cuda()
  validationset = validationset:cuda()
  model:cuda()
  criterion:cuda()
end

sgd_params = {
  learningRate = 1e-1
}

x, dl_dx = model:getParameters()

step = function(batch_size)
  local current_loss = 0
  local count = 0
  local shuffle = torch.randperm(Ntrain)
  batch_size = batch_size or 200

  for t = 1, Ntrain, batch_size do
    local size = math.min(t + batch_size - 1, Ntrain) - t
    local inputs = torch.Tensor(size, 28, 28)
    for i = 1,size do
      local input = trainset[shuffle[i+t]]
      inputs[i] = input
    end

    local feval = function(x_new)
      if x ~= x_new then x:copy(x_new) end
      dl_dx:zero()

      local loss = criterion:forward(model:forward(inputs), inputs)
      model:backward(inputs, criterion:backward(model.output, inputs))

      return loss, dl_dx
    end

    _, fs = optim.sgd(feval, x, sgd_params)
    count = count + 1
    current_loss = current_loss + fs[1]
  end

  return current_loss / count
end


eval = function(batch_size)
  local loss = 0
  local count = 0
  batch_size = batch_size or 200

  for i = 1, Nvalid, batch_size do
    local size = math.min(i + batch_size - 1, Nvalid) - i
    local inputs = validationset[{{i, i + size - 1}}]
    local current_loss = criterion:forward(model:forward(inputs), inputs)
    loss = loss + current_loss
    count = count + 1
  end

  return loss / count
end

max_iters = 5

train = function()
  local last_error = 0
  local increasing = 0
  local threshold = 1
  for i = 1, max_iters do
    local loss = step()
    print(string.format('Epoch: %d current loss: %4f', i, loss))
    local error = eval()
    print(string.format('Error on the validation set: %4f', error))
    if error > last_error then
      if increasing > threshold then break end
      increasing = increasing + 1
    else
      increasing = 0
    end
    last_error = error
  end
end

-- saving model
local path = require 'paths'
modelName = 'unsup_mnist.net'
filename = '/Users/art/torch_projects/unsup_project/' .. modelName

if path.filep(filename) then
  print("Model exists!")
  model = torch.load(filename)
else
  print("Model does not exist! Needs to be trained first.")
  train()
  torch.save(filename, model)
end

-- if itorch then
--   itorch.image(model:get(2).weight)
-- end

-- print(model:get(2).weight[{{}, {60, 70}}]:size())

eweight = model:get(1):get(2).weight
dweight = model:get(2):get(1).weight
-- print(eweight:transpose(1,2):unfold(2,32,32):size())
dweight = dweight:transpose(1,2):unfold(2,28,28)
eweight = eweight:unfold(2,28,28)
print(dweight:size())
print(eweight:size())
dd = image.toDisplayTensor{input=dweight,
                           padding=2,
                           nrow=math.floor(math.sqrt(28*28)),
                           symmetric=true}
de = image.toDisplayTensor{input=eweight,
                           padding=2,
                           nrow=math.floor(math.sqrt(28*28)),
                           symmetric=true}
if itorch then
  print('Decoder filters')
  itorch.image(dd)
  print('Encoder filters')
  itorch.image(de)
else
  print('run in itorch for visualization')
end

print(path.cwd())

image.save(path.cwd() .. '/filters_dec.jpg', dd)
image.save(path.cwd() .. '/filters_enc.jpg', de)
-- print(model:get(2).weight)
-- print(model:get(4))
-- print(model)
