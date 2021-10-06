"""
    SumOfSquaresError(NeuralNetwork)

Sum of squares error loss function.

    Usage:
```julia-repl
SumOfSquaresError(NeuralNetwork)
```
Arguments:
- `NeuralNetwork` : The NeuralNetwork structure.
"""
function SumOfSquaresError(NeuralNetwork)
	loss = 0.0

	for i in 1:size(NeuralNetwork.y, 1)
		loss += (NeuralNetwork.y[i] - NeuralNetwork.output[i]) ^ 2.0
    end
	
    return loss
end