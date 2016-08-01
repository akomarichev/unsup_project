local torch = require 'torch'
local optim = require 'optim'
local image = require 'image'
local cuda = pcall(require, 'cutorch')
local mnist = require 'mnist'
local path = require 'paths'

print('Setting up')
torch.setheaptracking(true)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(1)
if cuda then
  require 'cunn'
  cutorch.manualSeed(torch.random())
end

local cmd = torch.CmdLine()
cmd:option('-learningRate', 0.1, 'Learning rate')
cmd:option('-sparsity', 25, 'K-sparsity')
cmd:option('-epochs', 10, 'Training epochs')
local opt = cmd:parse(arg)

local testset = mnist.testdataset().data:float():div(255)

local trainset = mnist.traindataset().data[{{1, 50000}}]:float():div(255)

local validationset = mnist.traindataset().data[{{50001, 60000}}]:float():div(255)

local Ntrain = trainset:size(1)
local Ntest = testset:size(1)
local Nvalid = validationset:size(1)

if cuda then
  trainset = trainset:cuda()
  validationset = validationset:cuda()
end

print(type(trainset))
print(trainset:size())

local AE = require 'model'
AE:createAutoencoder(trainset, opt.sparsity)
local model = AE.autoencoder

if cuda then
  model:cuda()
end

local criterion = nn.BCECriterion()

if cuda then
  criterion:cuda()
end

local sgd_params = {
  learningRate = opt.learningRate
}

theta, dl_dx = model:getParameters()

local x

local feval = function(params)
  if theta ~= params then theta:copy(params) end
  dl_dx:zero()

  local Xhat = model:forward(x)
  local loss = criterion:forward(Xhat, x)
  local gradLoss = criterion:backward(Xhat, x)
  model:backward(x, gradLoss)

  return loss, dl_dx
end

local __, loss

step = function(batch_size)
  local current_loss = 0
  local count = 0
  batch_size = batch_size or 200

  for t = 1, Ntrain, batch_size do
    x = trainset:narrow(1, t, batch_size)

    __, loss = optim.sgd(feval, theta, sgd_params)
    count = count + 1
    current_loss = current_loss + loss[1]
  end

  return current_loss / count
end


eval = function(batch_size)
  local count = 0
  local l = 0
  batch_size = batch_size or 200

  for i = 1, Nvalid, batch_size do
    x = validationset:narrow(1, i, batch_size)
    local Xhat = model:forward(x)
    local current_loss = criterion:forward(Xhat, x)
    l = l + current_loss
    count = count + 1
  end

  return l / count
end

train = function()
  local last_error = 0
  local increasing = 0
  local threshold = 1
  for i = 1, opt.epochs do
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
modelName = '/unsup_mnist.net'
filename = path.cwd() .. modelName

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

image.save(path.cwd() .. '/filters_dec.jpg', dd)
image.save(path.cwd() .. '/filters_enc.jpg', de)
-- print(model:get(2).weight)
-- print(model:get(4))
-- print(model)
