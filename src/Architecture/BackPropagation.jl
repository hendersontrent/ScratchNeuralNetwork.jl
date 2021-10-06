"""
    BackPropagation(NeuralNetwork)

Backpropagation process for the neural network.

    Usage:
```julia-repl
BackPropagation(NeuralNetwork)
```
Arguments:
- `NeuralNetwork` : The NeuralNetwork structure.
"""
function BackPropogation(NeuralNetwork)
    
    # Use the chain rule to find derivative of loss function w.r.t. w₁ and w₂ (weights)

    ∂w₂ = transpose(NeuralNetwork.layer1) * ((2 * (NeuralNetwork.y - NeuralNetwork.output)) * SigmoidDerivative(NeuralNetwork.output))

    ∂w₁ = ((2 * (NeuralNetwork.y - NeuralNetwork.output)) * SigmoidDerivative(NeuralNetwork.output)) * transpose(NeuralNetwork.w₂)

    ∂w₁ = ∂w₁ * SigmoidDerivative(NeuralNetwork.layer1)
    
    ∂w₁ = transpose(NeuralNetwork.input) * ∂w₁

    # Update weights using gradient of the loss function

    NeuralNetwork$w₁ <- NeuralNetwork$w₁ + ∂w₁
    NeuralNetwork$w₂ <- NeuralNetwork$w₂ + ∂w₂

    return NeuralNetwork
end