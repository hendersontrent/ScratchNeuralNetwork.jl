"""
    MeanSquareError(NeuralNetwork)

Sum of square error loss function.

    Usage:
```julia-repl
MeanSquareError(NeuralNetwork)
```
Arguments:
- `NeuralNetwork` : The NeuralNetwork structure.
"""
function MeanSquareError(NeuralNetwork)
	ΣSE = 0.0

	for i in 1:size(NeuralNetwork.y, 1)
		ΣSE += (NeuralNetwork.y[i] - NeuralNetwork.output[i]) ^ 2.0
    end
	
    loss = 1.0 / size(NeuralNetwork.y) * ΣSE
    return loss
end