require 'nn'

local JoinGrid, Parent = torch.class('nn.JoinGrid', 'nn.Module')

function JoinGrid:__init(patch_size, numberOfPatches)
   Parent.__init(self)
   self.size = torch.LongStorage()
   self.patch_size = patch_size
   self.numberOfPatches = numberOfPatches
end

function JoinGrid:updateOutput(input)

   -- define the size of the joined table
   self.output:resize(input[1]:size(1), self.patch_size * self.numberOfPatches, self.patch_size * self.numberOfPatches)

   -- join grid
   local patchNumber = 1
   for x = 1, self.numberOfPatches do
     for y = 1, self.numberOfPatches do
       local currentInput = input[patchNumber]
       self.output[{ {}, {(x-1) * self.patch_size + 1, x * self.patch_size}, {(y-1) * self.patch_size + 1, y * self.patch_size} }]:copy(currentInput)
       patchNumber = patchNumber + 1
     end
   end

   return self.output
end


function JoinGrid:updateGradInput(input, gradOutput)

   self.gradInput:resize(#input, input[1]:size(1), self.patch_size, self.patch_size)

   local currentOutput= {}
   local patchNumber = 1
   for x = 1, self.numberOfPatches do
     for y = 1, self.numberOfPatches do
       local currentOutput = input[patchNumber]
       local currentGradInput = gradOutput[{ {}, {(x-1) * self.patch_size + 1, x * self.patch_size}, {(y-1) * self.patch_size + 1, y * self.patch_size} }]
       self.gradInput[patchNumber]:copy(currentGradInput)
       patchNumber = patchNumber + 1
     end
   end

   return self.gradInput
end
