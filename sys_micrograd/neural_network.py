import random
from micrograd_engine import Value 

#  1 - Define a single neuron : Each neuron has the (weights + bias) corresponding to the number of inputs (layer before)
class Neuron:
    
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
    
    
    

# 2 - Define a layer : each layer is a list of neurons : where
#            - each neuron takes in the Num inputs, and 
#            - the layer itself consists of #nurons = number of outputs
class Layer:
    
    def __init__(self, num_input, num_output):
        self.neurons = [Neuron(num_input) for _ in range(num_output)] # this creates a num_input* numoutput matric where each num_input == neruron
    
    def __call__(self, x):
        outputs = [n(x) for n in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs # allows us to get a single value out if only one neuron
    
    def parameters(self):
        return [ neuron_param for neuron in self.neurons for neuron_param in neuron.parameters() ]



# 3 -  Multi layer Perceptron
class MLP:
    
    def __init__(self, num_inputs, num_outputs):
        sz = [num_inputs] + num_outputs # 
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(num_outputs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [ layer_params for layer in self.layers for layer_params in layer.parameters() ]


