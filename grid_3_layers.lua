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

  enc1 = nn.ParallelTable()
  for i = 1, numberOfPatches * numberOfPatches do
    branch = nn.Sequential()
    branch:add(nn.View(-1, featureSize))
    branch:add(nn.Linear(featureSize, featureSize))
    branch:add(nn.Ksparse(K))
    branch:add(nn.View(-1, patch_size, patch_size))
    enc1:add(branch)
  end
  self.encoder:add(enc1)
  self.encoder:add(nn.JoinGrid(patch_size, numberOfPatches))
  self.encoder:add(nn.SplitGrid(patch_size, numberOfPatches))

  enc2 = nn.ParallelTable()
  for i = 1, numberOfPatches * numberOfPatches do
    branch = nn.Sequential()
    branch:add(nn.View(-1, featureSize))
    branch:add(nn.Linear(featureSize, featureSize))
    branch:add(nn.Ksparse(K))
    branch:add(nn.View(-1, patch_size, patch_size))
    enc2:add(branch)
  end
  self.encoder:add(enc2)
  self.encoder:add(nn.JoinGrid(patch_size, numberOfPatches))
  self.encoder:add(nn.SplitGrid(patch_size, numberOfPatches))

  enc3 = nn.ParallelTable()
  for i = 1, numberOfPatches * numberOfPatches do
    branch = nn.Sequential()
    branch:add(nn.View(-1, featureSize))
    branch:add(nn.Linear(featureSize, featureSize))
    branch:add(nn.Ksparse(K))
    enc3:add(branch)
  end
  self.encoder:add(enc3)

  self.decoder = nn.Sequential()

  dec1 = nn.ParallelTable()
  for i = 1, numberOfPatches * numberOfPatches do
    branch = nn.Sequential()
    branch:add(nn.Linear(featureSize, featureSize))
    branch:add(nn.Sigmoid(true))
    branch:add(nn.View(-1, patch_size, patch_size))
    dec1:add(branch)
  end

  self.decoder:add(dec1)
  self.decoder:add(nn.JoinGrid(patch_size, numberOfPatches))
  self.decoder:add(nn.SplitGrid(patch_size, numberOfPatches))

  dec2 = nn.ParallelTable()
  for i = 1, numberOfPatches * numberOfPatches do
    branch = nn.Sequential()
    branch:add(nn.View(-1, featureSize))
    branch:add(nn.Linear(featureSize, featureSize))
    branch:add(nn.Sigmoid(true))
    branch:add(nn.View(-1, patch_size, patch_size))
    dec2:add(branch)
  end

  self.decoder:add(dec2)
  self.decoder:add(nn.JoinGrid(patch_size, numberOfPatches))
  self.decoder:add(nn.SplitGrid(patch_size, numberOfPatches))

  dec3 = nn.ParallelTable()
  for i = 1, numberOfPatches * numberOfPatches do
    branch = nn.Sequential()
    branch:add(nn.View(-1, featureSize))
    branch:add(nn.Linear(featureSize, featureSize))
    branch:add(nn.Sigmoid(true))
    branch:add(nn.View(-1, patch_size, patch_size))
    dec3:add(branch)
  end

  self.decoder:add(dec3)
  self.decoder:add(nn.JoinGrid(patch_size, numberOfPatches))

  -- Tied weights
  for i = 1, numberOfPatches * numberOfPatches do
    self.decoder:get(7):get(i):get(2).weight = self.encoder:get(2):get(i):get(2).weight:t()
    self.decoder:get(7):get(i):get(2).gradWeight = self.encoder:get(2):get(i):get(2).gradWeight:t()

    self.decoder:get(4):get(i):get(1).weight = self.encoder:get(5):get(i):get(2).weight:t()
    self.decoder:get(4):get(i):get(1).gradWeight = self.encoder:get(5):get(i):get(2).gradWeight:t()

    self.decoder:get(1):get(i):get(1).weight = self.encoder:get(8):get(i):get(2).weight:t()
    self.decoder:get(1):get(i):get(1).gradWeight = self.encoder:get(8):get(i):get(2).gradWeight:t()
  end

  self.autoencoder = nn.Sequential()
  self.autoencoder:add(self.encoder)
  self.autoencoder:add(self.decoder)

end

return AE
