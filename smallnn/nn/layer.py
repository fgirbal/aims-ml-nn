import torch
import numpy as np
from smallnn.nn.init import InitializationType, xavier_normal
from typing import Tuple, Dict


class Layer():
    """Base Layer class"""
    def __init__(self):
        self.has_trainable_params = False

    def get_layer_output_shape(self, input_shape: Tuple[int, int]):
        # given an input vector, output the output vector shape
        return input_shape


class TrainableLayer(Layer):
    """Base trainable Layer class"""
    def __init__(self, out_features):
        self.has_trainable_params = True
        self.out_features = out_features

    def initialize_parameters(self, input_shape: Tuple[int, int]):
        # given an input shape, return a vector with the parameter tensor
        raise NotImplementedError

    def update_parameters(self, parameter_dictionary: Dict):
        # given a parameter dictionary, update the right parameters
        raise NotImplementedError


class Linear(TrainableLayer):
    """Linear NN layer"""
    def __init__(self, out_features: int):
        super().__init__(out_features)
        # initialize weight and bias to zeros
        self._weight: torch.Tensor = None
        self._bias: torch.Tensor = None

    @property
    def weight(self) -> torch.tensor:
        return self._weight

    @weight.setter
    def weight(self, value: torch.tensor) -> torch.tensor:
        # it must be a tensor with the right shape
        assert value.shape[1] == self.out_features

        return self._weight
    
    @property
    def bias(self) -> torch.tensor:
        return self._bias

    @bias.setter
    def bias(self, value: torch.tensor) -> torch.tensor:
        # it must be a tensor with the right output features shape
        assert value.shape[1] == self.out_features
        
        return self._bias

    def __call__(self, input_vector: torch.tensor) -> torch.tensor:
        if self._weight is None or self._bias is None:
            raise ValueError(
                "weight and bias parameters have not been initialized"
            )

        return input_vector @ self.weight + self.bias

    def initialize_parameters(
            self,
            input_shape: Tuple[int, int],
            init_type=InitializationType.XAVIER_NORMAL
        ) -> Dict:
        # weight and bias
        weight_shape = (input_shape[1], self.out_features)
        bias_shape = (input_shape[0], self.out_features)

        # initialize values to zero
        self._weight = torch.zeros(weight_shape, requires_grad=True)
        self._bias = torch.zeros(bias_shape, requires_grad=True)

        # xavier normal initialization
        if init_type == InitializationType.XAVIER_NORMAL:
            xavier_normal(self._weight)
            xavier_normal(self._bias)

        # return references to the parameter vectors
        return self.parameters

    @property
    def parameters(self) -> Dict:
        if self._weight is None or self._bias is None:
            raise ValueError(
                "weight and bias parameters have not been initialized"
            )

        return {
            "W": self._weight,
            "b": self._bias
        }

    def update_parameters(self, parameter_dictionary: Dict):
        # weight and bias
        if "W" in parameter_dictionary:
            self.weight = parameter_dictionary["W"]

        if "b" in parameter_dictionary:
            self.weight = parameter_dictionary["b"]

    def get_layer_output_shape(self, input_shape) -> Tuple[int, int]:
        return input_shape[0], self.out_features


class Flatten(Layer):
    """ReLu layer"""
    def __call__(self, input_vector: torch.tensor) -> torch.tensor:
        return input_vector.flatten(start_dim=1)

    def get_layer_output_shape(self, input_shape):
        return (input_shape[0], np.prod(input_shape[1:]))


class ReLU(Layer):
    """ReLu layer"""
    def __call__(self, input_vector: torch.tensor) -> torch.tensor:
        return torch.maximum(torch.zeros(input_vector.shape), input_vector)


class Tanh(Layer):
    """Tanh layer"""
    def __call__(self, input_vector: torch.tensor) -> torch.tensor:
        return torch.tanh(input_vector)


class Softmax(Layer):
    """Softmax layer"""
    def __call__(self, input_vector: torch.tensor) -> torch.tensor:
        return torch.exp(input_vector) / torch.sum(torch.exp(input_vector))
