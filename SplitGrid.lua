require 'nn'

local SplitGrid, Parent = torch.class('nn.SplitGrid', 'nn.Module')

function SplitGrid:__init(patch_size, numberOfPatches)
   Parent.__init(self)
   self.patch_size = patch_size
   self.numberOfPatches = numberOfPatches
end

function SplitGrid:updateOutput(input)
  local currentOutput = input.new(self.numberOfPatches*self.numberOfPatches, input:size(1), self.patch_size, self.patch_size)
  local patchNumber = 1
  
   for x = 1, self.numberOfPatches do
     for y = 1, self.numberOfPatches do
       currentOutput[patchNumber] = input[{ {}, {(x-1) * self.patch_size + 1, x * self.patch_size}, {(y-1) * self.patch_size + 1, y * self.patch_size} }]
       patchNumber = patchNumber + 1
     end
   end
   self.output = currentOutput
   return self.output
end

function SplitGrid:updateGradInput(input, gradOutput)
   if self.gradInput then
      self.gradInput:resizeAs(input)

      local patchNumber = 1
      for x = 1, self.numberOfPatches do
        for y = 1, self.numberOfPatches do
          local currentGradInput = gradOutput[patchNumber]
          self.gradInput[{ {}, {(x-1) * self.patch_size + 1, x * self.patch_size}, {(y-1) * self.patch_size + 1, y * self.patch_size} }]:copy(currentGradInput)
          patchNumber = patchNumber + 1
        end
      end
   end
   return self.gradInput
end
