from copy import deepcopy

import torch
import torch.nn as nn

from models.layers.mlp_compensate_denselayer import MultiLayerCompensateDense as Compensation
from models.layers.mlp_compensate_denselayer_indepbias import MultiLayerCompensateDenseIndpBias as CompensationP


class CompensateDnn(nn.Module):
    def __init__(self, input_dimensions, hidden_layers_dimensions, activation_function_choice, number_of_outputs=1,
                 compensation_layer_sizes=[128], compensation_activation='relu', compensation_dropout=0.5,
                 compensation_layer_location=None, number_of_classes=1, nan_input=False, data_input_modulate=False,
                 plus=False):
        """This extends the basic dnn architecture by adding the compensation layer at the input side."""

        super(CompensateDnn, self).__init__()

        # Model parameters based on inputs
        self.input_dimensions = input_dimensions
        self.number_of_outputs = number_of_outputs
        self.number_of_classes = number_of_classes
        self.compensation_layer_location = compensation_layer_location
        self.plus = plus
        self.hidden_layers_dimensions = deepcopy(hidden_layers_dimensions)
        if nan_input:
            self.hidden_layers_dimensions.insert(0, 2 * sum(input_dimensions))
        else:
            self.hidden_layers_dimensions.insert(0, sum(input_dimensions))
        self.hidden_layers_dimensions.append(number_of_classes)
        self.nan_input = nan_input
        number_of_hidden_layers = len(self.hidden_layers_dimensions) - 1
        input_layer_dimension = sum(input_dimensions)
        activation_function = {
            'relu': nn.ReLU,
            'elu': nn.ELU,
            'leaky_relu': nn.LeakyReLU,
            'tanh': nn.Tanh
        }
        self.activations = activation_function[activation_function_choice]
        self.input_activation = self.activations()

        self.data_input_modulate = data_input_modulate
        if data_input_modulate:
            modulate_input_dimension = input_layer_dimension * 2
        else:
            modulate_input_dimension = input_layer_dimension

        # Hidden layers
        self.hidden_layer_stream = nn.ModuleList()
        for layer in range(number_of_hidden_layers):
            if layer == compensation_layer_location:
                if plus:  # plus adds a parallel stream of fixed weights not discussed in paper (lower performance)
                    parallel_streams = nn.ModuleList()
                    parallel_streams.append(CompensationP(self.hidden_layers_dimensions[layer],
                                                          compensation_layer_sizes,
                                                          compensation_activation,
                                                          compensation_dropout,
                                                          compensation_layer_output_size=
                                                          self.hidden_layers_dimensions[layer + 1],
                                                          nan_flag_dimension=
                                                          modulate_input_dimension))
                    parallel_streams.append(nn.Linear(self.hidden_layers_dimensions[layer],
                                                      self.hidden_layers_dimensions[layer + 1]))
                    self.hidden_layer_stream.append(parallel_streams)
                else:
                    self.hidden_layer_stream.append(Compensation(self.hidden_layers_dimensions[layer],
                                                                 compensation_layer_sizes,
                                                                 compensation_activation,
                                                                 compensation_dropout,
                                                                 compensation_layer_output_size=
                                                                 self.hidden_layers_dimensions[layer + 1],
                                                                 nan_flag_dimension=
                                                                 modulate_input_dimension,
                                                                 sample_missing=False))
            else:
                self.hidden_layer_stream.append(nn.Linear(self.hidden_layers_dimensions[layer],
                                                          self.hidden_layers_dimensions[layer + 1]))

    def forward(self, network_input):
        """The input is assumed to be divided into three categories, numerical, categorical and missing flags. They are
        simply concatenated together here to form the input to the network. I leave it this way for future operations
        that could be performed on each one of them and for possibility of addition of new ones."""

        numeric_input = network_input[0]
        categorical_input = network_input[1]
        missing_flags = network_input[2]

        input_features = torch.cat((numeric_input, categorical_input), 1)
        if self.nan_input:
            input_features = torch.cat((input_features, missing_flags), 1)
        if self.data_input_modulate:
            missing_flags = torch.cat((missing_flags, torch.cat((numeric_input, categorical_input), 1)), 1)
        x = input_features

        for layer_idx, layer in enumerate(self.hidden_layer_stream):
            if layer_idx == self.compensation_layer_location:
                if self.plus:
                    x_0 = layer[0]([x, missing_flags])
                    x_1 = layer[1](x)
                    x = x_0 + x_1
                else:
                    x = layer([x, missing_flags], self.training)
            else:
                x = layer(x)
            if layer_idx == len(self.hidden_layer_stream) - 1:
                if self.number_of_classes > 1:
                    outputs = nn.LogSoftmax(dim=1)(x)
                else:
                    outputs = torch.sigmoid(x)
            else:
                x = self.input_activation(x)

        return outputs
