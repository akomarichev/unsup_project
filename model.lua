local nn = require "nn"
local Ksparse = require "Ksparse"

local AE = {}

function AE:createAutoencoder(X, K)
  local featureSize = X:size(2) * X:size(3)

  self.encoder = nn.Sequential()
  self.encoder:add(nn.View(-1, featureSize))
  self.encoder:add(nn.Linear(featureSize, featureSize))
  -- self.encoder:add(nn.ReLU(true))
  self.encoder:add(nn.Ksparse(K))

  self.decoder = nn.Sequential()
  self.decoder:add(nn.Linear(featureSize, featureSize))
  self.decoder:add(nn.Sigmoid(true))
  self.decoder:add(nn.View(X:size(2), X:size(3)))

  self.decoder:get(1).weight = self.encoder:get(2).weight:t()
  self.decoder:get(1).gradWeight = self.encoder:get(2).gradWeight:t()

  self.autoencoder = nn.Sequential()
  self.autoencoder:add(self.encoder)
  self.autoencoder:add(self.decoder)

end

return AE
