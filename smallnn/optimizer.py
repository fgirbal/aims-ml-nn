import torch
from smallnn.nn.sequential import SequentialNN


class Optimizer():
    """Base Optimizer class"""
    def __init__(self, network: SequentialNN):
        super(Optimizer, self).__init__()
        self.network = network

    def zero_grad(self):
        """To avoid accumulation, reset the parameters
        """
        for parameter in self.network.parameters:
            # if there's a value, reset it
            if parameter.grad is not None:
                parameter.grad.zero_()
    
    def step(self):
        """Take a step in the optimization
        """
        raise NotImplemented


class SGD(Optimizer):
    """Stochastic gradient descent with momentum implementation"""
    def __init__(self, network: SequentialNN, lr: float, momentum: float=0.0):
        super().__init__(network)
        self.lr = lr
        self.momentum = momentum
        self.last_step = []

    def step(self):
        """In stochastic gradient descent, take a step in the negative
        gradient direction.
        """
        with torch.no_grad():
            for i, parameter in enumerate(self.network.parameters):
                # compute the step
                step = parameter.grad

                # if there's a previous step, use it
                if len(self.last_step) == len(self.network.parameters):
                    step += self.momentum * self.last_step[i]
                    self.last_step[i] = step.detach().clone()
                else:
                    self.last_step.append(step)
                
                # update the parameter
                parameter -= self.lr * step

        