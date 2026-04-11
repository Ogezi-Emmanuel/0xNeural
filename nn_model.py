import math
import random


class Value:
    """
    This is the core of our engine. Every number in our neural network
    will be wrapped in this Value object so we can track its gradients.
    """
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0 # Initially, the gradient (derivative) is zero
        self._backward = lambda: None # Function to do the actual chain rule
        self._prev = set(_children) # Keeps track of the nodes that created this one
        self._op = _op # The operation that created this node (+, *, etc.)
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        # Allow adding a Value object and a regular Python number
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # The derivative of addition is 1.
            # Chain rule: local derivative (1) * global derivative (out.grad)
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # The derivative of multiplication is the *other* value.
            # Chain rule: local derivative (other.data) * global derivative (out.grad)
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        # Activation function: $f(x) = max(0, x)$
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            # The derivative of ReLU is 1 if x > 0, else 0.
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        # math.tanh is safe and won't overflow
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # The derivative of tanh is (1 - tanh^2)
            self.grad += (1.0 - t**2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        # This function kicks off the actual backpropagation.
        topo = []
        visited = set()

        # Build a topological graph so we process nodes in the exact reverse order
        # they were created.
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # The base case: the derivative of the final output with respect to itself is 1
        self.grad = 1.0

        # Go backwards through the topological graph and apply the chain rule
        for node in reversed(topo):
            node._backward()


class Neuron:
    def __init__(self, nin, nonlin=True):
        # THE FIX: Xavier Initialization
        # We scale down the random weights based on how many inputs this neuron has.
        scale = math.sqrt(nin)

        self.w = [Value(random.uniform(-1, 1) / scale) for _ in range(nin)]
        # Initialize bias to 0 to be extra safe
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        # Squeeze the output between -1 and 1 to prevent explosions
        return act.tanh() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=(i!=len(nouts)-1)) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
