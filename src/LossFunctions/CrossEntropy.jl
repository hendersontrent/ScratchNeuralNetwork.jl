"""
    CrossEntropy(NeuralNetwork)

Cross entropy loss function.

    Usage:
```julia-repl
CrossEntropy(NeuralNetwork)
```
Arguments:
- `NeuralNetwork` : The NeuralNetwork structure.
"""
function CrossEntropy(NeuralNetwork)
    ΣScore = 0.0

	for i in 1:size(NeuralNetwork.y, 1)
        ΣScore += NeuralNetwork.y[i] * log(1e-15 + NeuralNetwork.output[i])
    end

    MeanΣScore = 1.0 / size(NeuralNetwork.y, 1) * ΣScore
    loss = -MeanΣScore
	return loss
end