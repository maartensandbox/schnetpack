__all__ = ["KLCriterium"]

import torch
import torch.nn

# modifier to the cost function
class KLCriterium(torch.nn.Module):
    def __init__(self,**args):
        super(KLCriterium, self).__init__()
        pass

    def forward(self,output,target):        
        return output/len(target)
        