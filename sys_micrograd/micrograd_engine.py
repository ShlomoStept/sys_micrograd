import math
import numpy as np
import matplotlib.pyplot as plt


class Value:
    
    def __init__(self, data, _children=(), _op=(), label=''):
        self.data  = data
        self.grad = 0.0 
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    
    
    #  PART 1 -- Mathematical Operations
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value( self.data + other.data , (self, other), '+') 
        
        def _backward(): # by addition our job is to propogate __add__'s out's grad into self and other 
                         # i.e pass it backward (the out.grad backwards to the 2 parents) other and self
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value( self.data * other.data, (self, other), '*' ) 
          
        def _backward(): # by multiplication our job is to multiply the weight of the other variable (if we are talking about self we need other, if its other we need self)
                            # and we multiply this (ther other variables data) by the out.grad
                            # this is 
            self.grad  += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward    
    
        return out
    
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "We only support int or float data types for now"
        out = Value( self.data ** other, (self,), f'**{other}' ) 
        
        def _backward(): 
            self.grad += (other * ( self.data ** (other-1)) )* out.grad # other = n, and self.data = n
        out._backward = _backward
        
        return out
    
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        
        def _backward():
            self.grad += math.exp(x) * out.grad
        out._backward = _backward
        
        return out
           
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        
        
        def _backward(): # this is simply the derivative of the tanh function, multiplied by the out.grad 
                            # since this is the chain rule 
            self.grad = (1 - t**2) * out.grad
        out._backward = _backward
        
        return out
    
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self): 
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other): # other-self
        return other + (-self)
    
    def __truediv__(self, other): # self/other
        return self * other**-1
    
    def __rtruediv__(self, other): # other/self
        return other * self**-1
        
    def __rmul__(self, other): # for the case of -- other * self
        return self * other
    
    
    
    
    def backward(self):
        topo = []
        visited = set()
        # only adds a node once that nodes the children are added to the list
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        topo
        self.grad = 1.0
        # now we call the 
        for node in reversed(topo):
            node._backward()
    

    



if __name__ == '__main__':
    # inputs of the network : x1, x2
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')

    # weights of the network : w1, w2 
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')

    # bias of the neuron
    b = Value(6.88137335870195432, label='b')

    x1w1 = x1*w1 ; x1w1.label = "x1*w1";
    x2w2 = x2*w2 ;  x2w2.label = "x2*w2";
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'

    n = x1w1x2w2 + b
    n.label = 'n' # neuron before activation function

    #-----------
    e = (2*n).exp(); e.label = 'e'
    o = (e - 1)/(e + 1); o.label = 'o'
    #-----------
    o.label = 'o'
    o.backward()