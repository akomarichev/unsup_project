local nn = require "nn"
local Ksparse = require "Ksparse"
local SplitGrid = require 'SplitGrid'
local JoinGrid = require 'JoinGrid'

local AE = {}

function AE:createAutoencoder(X, K, patch_size)
  local featureSize = patch_size * patch_size
  local numberOfPatches = math.floor(X:size(2) / patch_size)

  self.encoder = nn.Sequential()
  self.encoder:add(nn.SplitGrid(patch_size, numberOfPatches))

  enc = nn.ParallelTable()
  for i = 1, numberOfPatches * numberOfPatches do
    branch = nn.Sequential()
    branch:add(nn.View(-1, featureSize))
    branch:add(nn.Linear(featureSize, featureSize))
    branch:add(nn.Ksparse(K))
    enc:add(branch)
  end
  self.encoder:add(enc)

  self.decoder = nn.Sequential()

  dec = nn.ParallelTable()
  for i = 1, numberOfPatches * numberOfPatches do
    branch = nn.Sequential()
    branch:add(nn.Linear(featureSize, featureSize))
    branch:add(nn.Sigmoid(true))
    branch:add(nn.View(-1, patch_size, patch_size))
    dec:add(branch)
  end

  self.decoder:add(dec)
  self.decoder:add(nn.JoinGrid(patch_size, numberOfPatches))

  -- Tied weights
  for i = 1, numberOfPatches * numberOfPatches do
    self.decoder:get(1):get(i):get(1).weight = self.encoder:get(2):get(i):get(2).weight:t()
    self.decoder:get(1):get(i):get(1).gradWeight = self.encoder:get(2):get(i):get(2).gradWeight:t()
  end

  self.autoencoder = nn.Sequential()
  self.autoencoder:add(self.encoder)
  self.autoencoder:add(self.decoder)

end

return AE
