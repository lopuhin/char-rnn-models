import torch.cuda
from torch.autograd import Variable


cuda_is_available = torch.cuda.is_available()


def variable(x):
    return cuda(Variable(x))


def cuda(x):
    return x.cuda() if cuda_is_available else x
