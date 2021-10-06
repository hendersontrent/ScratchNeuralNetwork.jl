"""
    SumOfSquaresError(NeuralNetwork)

Sum of square error loss function.

    Usage:
```julia-repl
SumOfSquaresError(NeuralNetwork)
```
Arguments:
- `NeuralNetwork` : The NeuralNetwork structure.
"""
function SumOfSquaresError(NeuralNetwork)
    loss = sum((NeuralNetwork.y - NeuralNetwork.output) ^ 2)
    return loss
end