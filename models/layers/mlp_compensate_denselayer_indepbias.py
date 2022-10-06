import torch
import torch.nn.functional as F
import torch.nn as nn


class MultiLayerCompensateDenseIndpBias(nn.Module):
    def __init__(self, input_dimension, hidden_layers_dimensions, activation_function_choice, dropout_rate=0.5,
                 compensation_layer_output_size=None, nan_flag_dimension=None):
        """A compensation layer where missing values cause boosting the weights of other inputs."""

        super(MultiLayerCompensateDenseIndpBias, self).__init__()

        # Model parameters based on inputs
        if nan_flag_dimension is None:
            self.nan_flag_dimension = input_dimension
        else:
            self.nan_flag_dimension = nan_flag_dimension
        self.input_dimension = input_dimension
        if compensation_layer_output_size is None:
            output_dimension = input_dimension
        else:
            output_dimension = compensation_layer_output_size
        self.output_dimension = output_dimension
        number_of_hidden_layers = len(hidden_layers_dimensions) - 1
        # Hidden layers
        self.hidden_layer_stream = nn.ModuleList()
        for layer in range(number_of_hidden_layers):
            self.hidden_layer_stream.append(nn.Linear(hidden_layers_dimensions[layer],
                                                      hidden_layers_dimensions[layer + 1]))
        activation_function = {
            'relu': nn.ReLU,
            'elu': nn.ELU,
            'leaky_relu': nn.LeakyReLU,
            'tanh': nn.Tanh
        }
        self.activations = activation_function[activation_function_choice]
        # Output layer (linear since it provides weights)
        self.output_branch_dense = nn.Linear(hidden_layers_dimensions[-1],
                                             input_dimension*output_dimension,
                                             bias=False)
        # self.output_branch_bias = nn.Linear(hidden_layers_dimensions[-1], output_dimension)

        self.input_layer = nn.Linear(self.nan_flag_dimension, hidden_layers_dimensions[0])
        self.input_activation = self.activations()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, layer_input):
        """The input is assumed to be divided into two items, actual values and missing values flags (1=missing).
        They must have exactly the same size. Also, missing values must be imputed with zeros for this to work
        properly."""

        input_values = layer_input[0]
        nan_flag = layer_input[1]
        # assert input_values.size() == nan_flag.size()

        input_features = self.input_layer(nan_flag)
        x = self.dropout(self.input_activation(input_features))

        for layer in self.hidden_layer_stream:
            x = self.input_activation(layer(x))
        compensation_weights_dense = self.output_branch_dense(x)
        # compensation_weights_bias = self.output_branch_bias(x)
        # print(input_values.shape)
        outputs = torch.bmm(input_values.view(input_values.shape[0],
                                              1,
                                              input_values.shape[1]),
                            compensation_weights_dense.view((-1,
                                                             self.input_dimension,
                                                             self.output_dimension))
                            )

        return outputs.squeeze(dim=1)
