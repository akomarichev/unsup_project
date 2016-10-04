local torch = require 'torch'
local optim = require 'optim'
local image = require 'image'
local cuda = pcall(require, 'cutorch')
local mnist = require 'mnist'
local path = require 'paths'
local cnn = require 'cnn_model'

print('Setting up')
torch.setheaptracking(true)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(1)
if cuda then
  require 'cunn'
  cutorch.manualSeed(torch.random())
end

local cmd = torch.CmdLine()
cmd:option('-learningRate', 0.001, 'Learning rate')
cmd:option('-sparsity', 4, 'K-sparsity')
cmd:option('-epochs', 20, 'Training epochs')
cmd:option('-patch_size', 14, 'Patch size')
local opt = cmd:parse(arg)

local testset = mnist.testdataset().data:double():div(255)
local y_test = mnist.testdataset().label

local trainset = mnist.traindataset().data[{{1, 50000}}]:double():div(255)
local y_train = mnist.traindataset().label[{{1, 50000}}]:add(1)

local validationset = mnist.traindataset().data[{{50001, 60000}}]:double():div(255)
local y_valid = mnist.traindataset().label[{{50001, 60000}}]

local Ntrain = trainset:size(1)
-- local Ntest = testset:size(1)
-- local Nvalid = validationset:size(1)

if cuda then
  trainset = trainset:cuda()
  y_test = y_test:cuda()
  y_train = y_train:cuda()
  y_valid = y_valid:cuda()
  validationset = validationset:cuda()
  testset = testset:cuda()
end

-- print(type(trainset))
-- print(trainset:size())

modelName = '/grid_training.net'
filename = path.cwd() .. modelName

if path.filep(filename) then
  print("Model exists!")
  autoencoder = torch.load(filename)
end

ksparse_nodes, containter_nodes = autoencoder:findModules('nn.Ksparse')
-- print(autoencoder)

-- changing K parameter in original autoencoder (multiplied by 3)
for i = 1, #ksparse_nodes do
  ksparse_nodes[i].k = ksparse_nodes[i].k * 3
end

model = cnn_model(trainset, autoencoder:get(1), opt.patch_size)
-- print(autoencoder)
-- print(model)

if cuda then
  model:cuda()
end

local criterion = nn.ClassNLLCriterion()

if cuda then
  criterion:cuda()
end

local sgd_params = {
  learningRate = opt.learningRate
}

theta, dl_dx = model:getParameters()

local x
local y

local feval = function(params)
  if theta ~= params then theta:copy(params) end
  dl_dx:zero()

  local pred = model:forward(x)
  local loss = criterion:forward(pred, y)
  local gradLoss = criterion:backward(pred, y)
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
    y = y_train:narrow(1, t, batch_size)
    -- y:add(1)

    __, loss = optim.adam(feval, theta, sgd_params)
    count = count + 1
    current_loss = current_loss + loss[1]
  end

  return current_loss / count
end


eval = function(dataset, targets, batch_size)
  local count = 0
  local l = 0
  batch_size = batch_size or 200

  for i = 1, dataset:size(1), batch_size do
    x = dataset:narrow(1, i, batch_size)
    y = targets:narrow(1, i, batch_size)
    if cuda then
      y:cuda()
    else
      y:long()
    end
    local pred = model:forward(x)
    local _, indices = torch.max(pred, 2)
    indices:add(-1)
    local right_predictions = indices:eq(y):sum()
    count = count + right_predictions
  end

  return count / dataset:size(1)
end

train = function()
  local last_accuracy= 0
  local decreasing = 0
  local threshold = 1
  for i = 1, opt.epochs do
    local loss = step()
    print(string.format('Epoch: %d current loss: %4f', i, loss))
    local accuracy = eval(validationset, y_valid)
    print(string.format('Error on the validation set: %4f', accuracy))
    if accuracy < last_accuracy then
      if decreasing > threshold then break end
      decreasing = decreasing + 1
    else
      decreasing = 0
    end
    last_accuracy = accuracy
  end
end

train()
print('test accuracy: ' .. eval(testset, y_test))
--
-- -- print(model)
--
-- -- saving model
-- modelName = '/grid_training.net'
-- filename = path.cwd() .. modelName
--
-- if path.filep(filename) then
--   print("Model exists!")
--   model = torch.load(filename)
-- else
--   print("Model does not exist! Needs to be trained first.")
--   train()
--   torch.save(filename, model)
-- end
--
-- if itorch then
--   itorch.image(model:get(2).weight)
-- end
--
-- local numberOfPatches = math.floor(trainset:size(2) / opt.patch_size)
--
-- for i = 1, numberOfPatches * numberOfPatches do
--   eweight = model:get(1):get(2):get(i):get(2).weight
--   dweight = model:get(2):get(1):get(i):get(1).weight
--   dweight = dweight:transpose(1,2):unfold(2, opt.patch_size, opt.patch_size)
--   eweight = eweight:unfold(2, opt.patch_size, opt.patch_size)
--
--   dd = image.toDisplayTensor{input=dweight,
--                              padding=2,
--                              nrow=math.floor(math.sqrt(opt.patch_size * opt.patch_size)),
--                              symmetric=true}
--   de = image.toDisplayTensor{input=eweight,
--                              padding=2,
--                              nrow=math.floor(math.sqrt(opt.patch_size * opt.patch_size)),
--                              symmetric=true}
--
--   if itorch then
--     print('Decoder filters')
--     itorch.image(dd)
--     print('Encoder filters')
--     itorch.image(de)
--   else
--     print('run in itorch for visualization')
--   end
--
--   image.save(path.cwd() .. '/grid_decoder/filters_dec' .. i .. '.jpg', dd)
--   image.save(path.cwd() .. '/grid_encoder/filters_enc' .. i .. '.jpg', de)
-- end
