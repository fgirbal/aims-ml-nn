import torch
from typing import List, Tuple, Type
from smallnn.nn.layer import Layer


class SequentialNN():
    """Sequential neural network class"""
    def __init__(self, layers: List[Type[Layer]], input_shape: Tuple[int, int]):
        self.layers = layers

        # obtain trainable parameters vector
        input_to_layer_shape = input_shape

        # initialize the layers with parameters
        for layer in layers:
            # if there are parameters, initialize them
            if layer.has_trainable_params:
                params_of_layer = layer.initialize_parameters(
                    input_to_layer_shape
                )

            # continue to determine the network's output shape
            input_to_layer_shape = layer.get_layer_output_shape(
                input_to_layer_shape
            )

        self.output_shape = input_to_layer_shape

    def __call__(self, input_vector: torch.tensor):
        return self.forward(input_vector)

    def forward(self, input_vector: torch.tensor):
        """Forward pass through the network
        
        Args:
            input_vector (torch.tensor): input to the network
        
        Returns:
            torch.tensor: output tensor of running the network
        """
        output = input_vector.float()
        for layer in self.layers:
            output = layer(output)

        return output

    @property
    def parameters(self) -> List[torch.tensor]:
        """Return list of tensors of parameters of the network.
        
        Returns:
            List[torch.tensor]: parameters of the network
        """
        return [
            tensor
            for layer in self.layers if layer.has_trainable_params
            for tensor in layer.parameters.values()
        ]
