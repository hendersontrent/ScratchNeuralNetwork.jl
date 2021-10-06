"""
    NeuralNetwork(X, y, w₁, w₂, output)

Creates a container to hold all the necessary components for the network that other package functions refer to.

    Usage:
```julia-repl
NeuralNetwork(X, y, w₁, w₂, output)
```
"""
struct NeuralNetwork
    X::Array
    y::Array
    w₁::Array
    w₂::Array
    output::Array
end