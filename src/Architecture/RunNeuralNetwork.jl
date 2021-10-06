"""
    RunNeuralNetwork(Epochs, ActivationFunction, LossFunction, NeuralNetwork)

Train a neural network.

    Usage:
```julia-repl
RunNeuralNetwork(epochs, ActivationFunction, LossFunction, NeuralNetwork)
```
Arguments:
- `Epochs` : The number of iterations to train the network for.
- `ActivationFunction` : The activation function to use.
- `LossFunction` : The loss function to use.
- `NeuralNetwork` : The NeuralNetwork structure.
"""
function RunNeuralNetwork(Epochs::Int64 = 1000, ActivationFunction::String = "Sigmoid", LossFunction::String = "MeanSquareError", NeuralNetwork)

    # Check function arguments

    (ActivationFunction == "Sigmoid" || ActivationFunction == "Tanh" || ActivationFunction == "ReLU" || ActivationFunction == "Linear") || error("`ActivationFunction` should be a String of either 'Sigmoid', 'Tanh', 'ReLU' or 'Linear'.")

    (LossFunction == "SumOfSquaresError" || LossFunction == "MeanSquareError" || LossFunction == "CrossEntropy") || error("`LossFunction` should be a String of either 'SumOfSquaresError', 'MeanSquareError', or 'CrossEntropy'.")

    # Instantiate empty vector to store loss function results

    loss = zeros(Epochs)

    # Train the network

    for i in 1:Epochs
        print(string("Running epoch: ", i, " of ", Epochs))
        NeuralNetwork = FeedForward(NeuralNetwork)
        NeuralNetwork = BackPropogation(NeuralNetwork, ActivationFunction)

        if LossFunction == "MeanSquareError"
            loss[i] = MeanSquareError(NeuralNetwork)
        elseif LossFunction == "SumOfSquaresError"
            loss[i] = SumOfSquaresError(NeuralNetwork)
        elseif LossFunction == "CrossEntropy"
            loss[i] = CrossEntropy(NeuralNetwork)
        end
    end
end