#----------------------------------------
# This script sets out to produce a basic
# neural network from scratch
#----------------------------------------

#----------------------------------------
# Author: Trent Henderson, 5 October 2021
#----------------------------------------

using LinearAlgebra, Plots

# Activation function and its derivative

function Sigmoid(x::Array)
    x₁ = 1.0 / (1.0 + exp(-x))
    return x₁
end

function SigmoidDerivative(x::Array)
    x₁ = x * (1.0 - x)
    return x₁
end

# Loss function

function SumOfSquaresError(NeuralNetwork)
    loss = sum((NeuralNetwork.y - NeuralNetwork.output) ^ 2)
    return loss
end

# Feed forward

function FeedForward(NeuralNetwork)
    NeuralNetwork.layer1 = sigmoid(NeuralNetwork.input %*% NeuralNetwork.w₁)
    NeuralNetwork.output = sigmoid(NeuralNetwork.layer1 %*% NeuralNetwork.w₂)
end

# Backpropagation

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

# Run the NeuralNetwork

function RunNeuralNetwork(epochs::Int64 = 1000, NeuralNetwork)

    # Instantiate empty vector to store loss function results

    loss = zeros(epochs)

    # Train the network

    for i in 1:epochs
        print(string("Running epoch: ", i, " of ", epochs))
        NeuralNetwork = FeedForward(NeuralNetwork)
        NeuralNetwork = BackPropogation(NeuralNetwork)
        loss[i] = SumOfSquaresError(NeuralNetwork)
    end
end
