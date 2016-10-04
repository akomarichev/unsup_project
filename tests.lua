require 'torch'
require 'nn'
local mnist = require 'mnist'
local Ksparse = require "Ksparse"
local SplitGrid = require 'SplitGrid'
local JoinGrid = require 'JoinGrid'

local mytester = torch.Tester()
local precision = 1e-5
local jac = nn.Jacobian
local nntest = torch.TestSuite()

-- torch.setdefaulttensortype('torch.DoubleTensor')

local trainset = mnist.traindataset().data[{{1, 5}}]:double():div(255)
local K = 3
local image_size = trainset:size(2)
local patch_size = 14

local numberOfPatches = math.floor(image_size / patch_size)
featureSize = patch_size * patch_size

-- torch.setheaptracking(true)

function nntest.Exp()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.Exp()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Linear()
  local ini = math.random(10,20)
  local inj = math.random(10,20)

  local input = torch.randn(ini)
  mlp = nn.Linear(ini, inj)

  local err = jac.testJacobian(mlp, input)
  mytester:assertlt(err,precision, 'error on state ')
end

-- function nntest.KSparse()
--   local ini = math.random(10,20)
--   local inj = math.random(10,20)
--   local input = torch.randn(ini, inj)
--
--   local mlp = nn.Ksparse(K)
--   mlp:type('torch.FloatTensor')
--
--   local err = jac.testJacobian(mlp, trainset)
--   mytester:assertlt(err,precision, 'error on state ')
-- end

-- function nntest.KSparseModel()
--   local AE = require 'model'
--   AE:createAutoencoder(trainset, K)
--   local mlp = AE.autoencoder
--
--   local err = jac.testJacobian(mlp, trainset)
--   mytester:assertlt(err, precision, 'error on state')
-- end

-- function nntest.GridModel()
--   local AE = require 'grid_model'
--   AE:createAutoencoder(trainset, K, patch_size)
--   local model = AE.autoencoder
--
--   local err = jac.testJacobian(model, trainset)
--   mytester:assertlt(err, precision, 'error on state')
-- end

-- function nntest.GridDeepModel()
--   local AE = require 'grid_deep_model'
--   AE:createAutoencoder(trainset, K, patch_size)
--   local model = AE.autoencoder
--
--   local err = jac.testJacobian(model, trainset)
--   mytester:assertlt(err, precision, 'error on state')
-- end

function nntest.GridDeepModel()
  local AE = require 'grid_3_layers'
  AE:createAutoencoder(trainset, K, patch_size)
  local model = AE.autoencoder
  -- print(model)

  local err = jac.testJacobian(model, trainset)
  mytester:assertlt(err, precision, 'error on state')
end


-- function nntest.CNNModel()
--   modelName = '/grid_training.net'
--   filename = path.cwd() .. modelName
--
--   if path.filep(filename) then
--     print("Model exists!")
--     autoencoder = torch.load(filename)
--   end
--
--   local model = cnn_model(trainset, autoencoder:get(1), opt.patch_size)
--
--   local err = jac.testJacobian(model, trainset)
--   mytester:assertlt(err, precision, 'error on state')
-- end


-- function nntest.GridSimpleModel()
--   mlp = nn.Sequential()
--   mlp:add(nn.SplitGrid(patch_size, numberOfPatches))
--   c = nn.ParallelTable()
--   for i = 1, numberOfPatches * numberOfPatches do
--     branch = nn.Sequential()
--     -- branch:add(nn.Identity())
--     branch:add(nn.View(-1, featureSize))
--     branch:add(nn.Linear(featureSize, featureSize))
--     branch:add(nn.Ksparse(K))
--     branch:add(nn.View(-1, patch_size, patch_size))
--     c:add(branch)
--   end
--   mlp:add(c)
--   mlp:add(nn.JoinGrid(patch_size, numberOfPatches))
--
--   local err = jac.testJacobian(mlp, trainset)
--   mytester:assertlt(err, precision, 'error on state')
-- end

-- function nntest.SimpleModel()
--   local AE = require 'model_simple'
--   AE:createAutoencoder(trainset)
--   local mlp = AE.autoencoder
--
--   local err = jac.testJacobian(mlp, trainset)
--   mytester:assertlt(err, precision, 'error on state')
-- end

mytester:add(nntest)
mytester:run()
