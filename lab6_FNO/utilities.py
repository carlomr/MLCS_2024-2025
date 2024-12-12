import torch

class LprelLoss(): 
    """ 
    Sum of relative errors in L^p norm 
    
    x, y: torch.tensor
          x and y are tensors of shape (n_samples, *n, d_u)
          where *n indicates that the spatial dimensions can be arbitrary
    """
    def __init__(self, p:int, size_mean=False):
        self.p = p
        self.size_mean = size_mean

    def rel(self, x, y):
        num_examples = x.size(0)
        
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), p=self.p, dim=1)
        y_norms = torch.norm(y.reshape(num_examples, -1), p=self.p, dim=1)
        
        # check division by zero
        if torch.any(y_norms <= 1e-5):
            raise ValueError("Division by zero")
        
        if self.size_mean is True:
            return torch.mean(diff_norms/y_norms)
        elif self.size_mean is False:
            return torch.sum(diff_norms/y_norms) # sum along batchsize
        elif self.size_mean is None:
            return diff_norms/y_norms # no reduction
        else:
            raise ValueError("size_mean must be a boolean or None")
    
    def __call__(self, x, y):
        return self.rel(x, y)