"""
    FeedForward(NeuralNetwork)

Feed forward process for the neural network.

    Usage:
```julia-repl
FeedForward(NeuralNetwork)
```
Arguments:
- `NeuralNetwork` : The NeuralNetwork structure.
"""
function FeedForward(NeuralNetwork)
    NeuralNetwork.layer1 = sigmoid(NeuralNetwork.input * NeuralNetwork.w₁)
    NeuralNetwork.output = sigmoid(NeuralNetwork.layer1 * NeuralNetwork.w₂)
end