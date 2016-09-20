local nn = require "nn"
local Ksparse = require "Ksparse"
local SplitGrid = require 'SplitGrid'
local JoinGrid = require 'JoinGrid'

function cnn_model(X, encoder, patch_size)
  local featureSize = X:size(2) * X:size(2)
  local numberOfPatches = math.floor(X:size(2) / patch_size)

  model = nn.Sequential()
  model:add(encoder)
  dec = nn.ParallelTable()
  for i = 1, numberOfPatches * numberOfPatches do
    branch = nn.Sequential()
    branch:add(nn.View(-1, patch_size, patch_size))
    dec:add(branch)
  end
  model:add(dec)
  model:add(nn.JoinGrid(patch_size, numberOfPatches))
  model:add(nn.View(-1, featureSize))
  model:add(nn.Linear(featureSize, featureSize))
  model:add(nn.ReLU())
  model:add(nn.Linear(featureSize, 10))
  model:add(nn.LogSoftMax())

  return model
end
