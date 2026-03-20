"""Neural network building blocks built on the micrograd autograd engine.

Provides Neuron, Layer, and MLP classes that compose Value objects into
trainable neural network architectures. All operations flow through the
autograd engine, enabling gradient computation via loss.backward().
"""

import random
from micrograd_engine import Value


class Neuron:
    """A single neuron with tanh activation.

    Computes tanh(sum(w_i * x_i) + b) where weights and bias are Value
    objects, enabling automatic gradient computation.

    Args:
        num_inputs: Number of input connections (dimension of input vector).
    """

    def __init__(self, num_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(num_inputs)]
        self.b = Value(random.uniform(-1, 1))
        
    def __call__(self, x ):
        # w * x + b
        #print(list(zip(self.w, x)))
        act =  sum( (wi*xi for wi, xi in zip(self.w, x)), self.b )
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]
    
    
    

class Layer:
    """A layer of independent neurons that all receive the same input.

    Args:
        num_input: Dimensionality of the input vector.
        num_output: Number of neurons in this layer (output dimension).
    """

    def __init__(self, num_input, num_output):
        self.neurons = [Neuron(num_input) for _ in range(num_output)] # this creates a num_input* numoutput matric where each num_input == neruron
    
    def __call__(self, x):
        outputs = [n(x) for n in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs # allows us to get a single value out if only one neuron
    
    def parameters(self):
        return [ neuron_param for neuron in self.neurons for neuron_param in neuron.parameters() ]



class MLP:
    """Multi-Layer Perceptron: sequential stack of Layer objects.

    Args:
        num_inputs: Dimensionality of the input.
        num_outputs: List of layer sizes, e.g. [4, 4, 1] for two hidden
            layers of 4 neurons and a single output neuron.

    Example:
        >>> model = MLP(3, [4, 4, 1])  # 3 inputs, 2 hidden layers, 1 output
        >>> model([1.0, 2.0, 3.0])     # returns a Value
    """

    def __init__(self, num_inputs, num_outputs):
        sz = [num_inputs] + num_outputs # 
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(num_outputs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [ layer_params for layer in self.layers for layer_params in layer.parameters() ]


