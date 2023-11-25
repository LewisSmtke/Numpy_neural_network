"""Module containing functions for a single forward and backwards pass"""
import activations as acti


def single_forward_pass(weight, bias, input_vector, activation: str):
    """
    Function that contains a single forward pass through one layer.
    The activation function can be specified via string, so that it can be switched between layers.
    """
    layer_output = weight.dot(input_vector) + bias

    if activation.lower() == "relu":
        activated_ouput = acti.relu(layer_output)

    elif activation.lower() == "softmax":
        activated_ouput = acti.softmax(layer_output)

    elif activation.lower() == "tanh":
        activated_ouput = acti.softmax(layer_output)

    elif activation.lower() == "sigmoid":
        activated_ouput = acti.sigmoid * layer_output

    return activated_ouput
