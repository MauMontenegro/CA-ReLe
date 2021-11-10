import torch.nn as nn


class BaseNet(nn.Module):
    name = "BaseNet"

    def __init__(self, config):
        self.discrete = True
        self.__device__ = None
        super(BaseNet, self).__init__()

    def new(self):
        return newNet(self, self.config)

    @property
    def device(self):
        if self.__device__ is None:
            self.__device__ = next(self.parameters()).device
        return self.__device__


# Functions to manage and create networks

def newNet(net, config={}):
    netClass = net.__class__
    if config == {}:
        config = net.config
    new = netClass(config)
    return new
