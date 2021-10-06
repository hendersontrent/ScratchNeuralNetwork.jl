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
function BackPropogation(NeuralNetwork, ActivationFunction::String = "Sigmoid")

    # Check activation function argument

    (ActivationFunction == "Sigmoid" || ActivationFunction == "Tanh" || ActivationFunction == "ReLU" || ActivationFunction == "Linear") || error("`ActivationFunction` should be a String of either 'Sigmoid', 'Tanh', 'ReLU' or 'Linear'.")
    
    # Use the chain rule to find derivative of loss function w.r.t. w₁ and w₂ (weights)

    if ActivationFunction == "Sigmoid"
        DerivOutput = SigmoidDerivative(NeuralNetwork.output)
        DerivLayer1 = SigmoidDerivative(NeuralNetwork.output)
    elseif ActivationFunction == "Tanh"
        DerivOutput = TanhDerivative(NeuralNetwork.output)
        DerivLayer1 = TanhDerivative(NeuralNetwork.output)
    elseif ActivationFunction == "ReLU"
        DerivOutput = ReLUDerivative(NeuralNetwork.output)
        DerivLayer1 = ReLUDerivative(NeuralNetwork.output)
    elseif ActivationFunction == "Linear"
        DerivOutput = LinearDerivative(NeuralNetwork.output)
        DerivLayer1 = LinearDerivative(NeuralNetwork.output)
    end

    ∂w₂ = transpose(NeuralNetwork.layer1) * ((2 * (NeuralNetwork.y - NeuralNetwork.output)) * DerivOutput)

    ∂w₁ = ((2 * (NeuralNetwork.y - NeuralNetwork.output)) * DerivOutput) * transpose(NeuralNetwork.w₂)

    ∂w₁ = ∂w₁ * DerivLayer1
    
    ∂w₁ = transpose(NeuralNetwork.input) * ∂w₁

    # Update weights using gradient of the loss function

    NeuralNetwork$w₁ <- NeuralNetwork$w₁ + ∂w₁
    NeuralNetwork$w₂ <- NeuralNetwork$w₂ + ∂w₂

    return NeuralNetwork
end