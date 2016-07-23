require 'torch'
require 'nn'
require 'optim'
local mnist = require 'mnist'

-- ds = dp.Mnist()
--
-- print(ds:featureSize())
-- print(ds:classes())
-- print(#(ds:classes()))
--
-- for k,v in ipairs(ds:classes()) do
--    print(k,v)
-- end

local Xtrain = mnist.traindataset() --.data:float():div(255)
local testset = mnist.testdataset()

testset.data = testset.data:double():div(255)

trainset = {
  size = 50000,
  data = Xtrain.data[{{1, 50000}}]:double():div(255), --:float():div(255),
  label = Xtrain.label[{{1, 50000}}]
}

print(trainset)

validationset = {
  size = 10000,
  data = Xtrain.data[{{50001, 60000}}]:double():div(255), --float():div(255),
  label = Xtrain.label[{{50001, 60000}}]
}

print(validationset)


-- Let's build a model

model = nn.Sequential()
model:add(nn.View(-1, 28 * 28))
model:add(nn.Linear(28 * 28, 30))
model:add(nn.Tanh())
model:add(nn.Linear(30, 10))
model:add(nn.LogSoftMax())


criterion = nn.ClassNLLCriterion()

sgd_params = {
  learningRate = 1e-2,
  learningRateDecay = 1e-4,
  weightDecay = 1e-3,
  momentum = 1e-4
}

x, dl_dx = model:getParameters()

step = function(batch_size)
  local current_loss = 0
  local count = 0
  local shuffle = torch.randperm(trainset.size)
  batch_size = batch_size or 200

  for t = 1, trainset.size, batch_size do
    local size = math.min(t + batch_size - 1, trainset.size) - t
    local inputs = torch.Tensor(size, 28, 28)
    local targets = torch.Tensor(size)
    for i = 1,size do
      local input = trainset.data[shuffle[i+t]]
      local target = trainset.label[shuffle[i+t]]
      inputs[i] = input
      targets[i] = target
    end
    targets:add(1)

    local feval = function(x_new)
      if x ~= x_new then x:copy(x_new) end
      dl_dx:zero()

      local loss = criterion:forward(model:forward(inputs), targets)
      model:backward(inputs, criterion:backward(model.output, targets))

      return loss, dl_dx
    end

    _, fs = optim.sgd(feval, x, sgd_params)
    count = count + 1
    current_loss = current_loss + fs[1]
  end

  return current_loss / count
end

eval = function(dataset, batch_size)
  local count = 0
  batch_size = batch_size or 200

  for i = 1, dataset.size, batch_size do
    local size = math.min(i + batch_size - 1, dataset.size) - i
    local inputs = dataset.data[{{i, i + size - 1}}]
    local targets = dataset.label[{{i, i + size - 1}}]:long()
    local outputs = model:forward(inputs)
    local _, indices = torch.max(outputs, 2)
    indices:add(-1)
    local guessed_right = indices:eq(targets):sum()
    count = count + guessed_right
  end

  return count / dataset.size
end

max_iters = 30

train = function()
  local last_accuracy = 0
  local decreasing = 0
  local threshold = 1
  for i = 1, max_iters do
    local loss = step()
    print(string.format('Epoch: %d current loss: %4f', i, loss))
    local accuracy = eval(validationset)
    print(string.format('Accuracy on the validation set: %4f', accuracy))
    if accuracy < last_accuracy then
      if decreasing > threshold then break end
      decreasing = decreasing + 1
    else
      decreasing = 0
    end
    last_accuracy = accuracy
  end
end

-- saving model
local path = require 'paths'
modelName = 'sup_mnist.net'
filename = '/Users/art/torch_projects/first_project/' .. modelName

if path.filep(filename) then
  print("Model exists!")
  model = torch.load(filename)
else
  print("Model does not exist! Needs to be trained first.")
  train()
  torch.save(filename, model)
end

if itorch then
  itorch.image(model:get(2).weight[{{}, {60, 70}}])
end

print(model:get(2).weight[{{}, {60, 70}}]:size())
print(model:size())
-- print(model:get(2).weight)
-- print(model:get(4))
print(model)

-- print(model:get(2).weight)

print(string.format('Accuracy: %4f', eval(testset)))
