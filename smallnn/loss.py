import torch
from smallnn.nn.sequential import SequentialNN

def categorial_crossentropy(network: SequentialNN, input_vector: torch.tensor, targets: torch.tensor) -> torch.tensor:
    """Calculates the categorical cross entropy of the softmaxed output
    
    Args:
        network (SequentialNN): the network in question
        input_vector (torch.tensor): input to the network
        targets (torch.tensor): groundtruth outputs
    
    Returns:
        torch.tensor: tensor corresponding to the cross entropy between the
        softmaxed output and the targets
    """
    # forward pass to obtain the predictions
    predictions = network(input_vector)

    # use the logsumexp trick to compute the log(softmax(predictions))
    log_softmax = predictions - torch.logsumexp(predictions, axis=1, keepdims=True)

    # turn targets into a one-hot encoding
    targets_onehot = torch.zeros(log_softmax.shape)
    targets_onehot.scatter_(1, targets.view(-1, 1), 1)

    # calculate the negative log likelihood of the batch
    return -torch.sum(log_softmax * targets_onehot) / log_softmax.shape[0]


def l2_parameter_norm(network: SequentialNN) -> torch.tensor:
    """Calculates the L2 norm of the parameters of the network for
    regularization purposes.
    
    Args:
        network (SequentialNN): the network in question
    
    Returns:
        torch.tensor: tensor corresponding to the L2 norm of the parameters
        of the network
    """
    all_parameters = torch.cat([tensor.flatten() for tensor in network.parameters])
    return l2_norm(all_parameters)


def l2_norm(tensor: torch.tensor) -> torch.tensor:
    """Returns the L2 norm of a tensor
    
    Args:
        tensor (torch.tensor): input tensor
    
    Returns:
        torch.tensor: tensor corresponding to the L2 norm of the input
    """
    if len(tensor.shape) > 1:
        tensor = tensor.flatten()

    return torch.dot(tensor, tensor)
