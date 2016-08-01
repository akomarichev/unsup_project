require 'nn'

local Ksparse, Parent = torch.class('nn.Ksparse', 'nn.Module')

function Ksparse:__init(k)
   Parent.__init(self)
   self.k = k
   self.mask = torch.Tensor()
end

function Ksparse:updateOutput(input)
   self.mask:resizeAs(input):zero()
   self.output:resizeAs(input):copy(input)
   N = self.output:size(1)
   res, ind = self.output:topk(self.k, true)
   for i = 1, N do
     self.mask[i]:indexFill(1, ind[i], 1)
   end
   self.output:cmul(self.mask)
   return self.output
end

function Ksparse:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   self.gradInput:cmul(self.mask)
   return self.gradInput
end
