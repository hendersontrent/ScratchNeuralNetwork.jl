"""
    RunNeuralNetwork(Epochs, ActivationFunction, LossFunction, EarlyStopping, Patience, NeuralNetwork)

Train a neural network.

    Usage:
```julia-repl
RunNeuralNetwork(epochs, ActivationFunction, LossFunction, EarlyStopping, Patience, NeuralNetwork)
```
Arguments:
- `Epochs` : The number of iterations to train the network for.
- `ActivationFunction` : The activation function to use.
- `LossFunction` : The loss function to use.
- `EarlyStopping` : Whether to stop training early based on Patience.
- `Patience` : The number of epochs to assess improvement in loss reduction over.
- `NeuralNetwork` : The NeuralNetwork structure.
"""
function RunNeuralNetwork(Epochs::Int64 = 1000, ActivationFunction::String = "Sigmoid", LossFunction::String = "MeanSquareError", EarlyStopping::Bool = false, Patience::Int64 = 10, NeuralNetwork)

    # Check function arguments

    (ActivationFunction == "Sigmoid" || ActivationFunction == "Tanh" || ActivationFunction == "ReLU" || ActivationFunction == "Linear") || error("`ActivationFunction` should be a String of either 'Sigmoid', 'Tanh', 'ReLU' or 'Linear'.")

    (LossFunction == "SumOfSquaresError" || LossFunction == "MeanSquareError" || LossFunction == "CrossEntropy") || error("`LossFunction` should be a String of either 'SumOfSquaresError', 'MeanSquareError', or 'CrossEntropy'.")

    # Instantiate empty vector to store loss function results

    loss = zeros(Epochs)

    # Instantiate early stopping counter

    BadCounter = 0

    # Train the network

    for i in 1:Epochs
        NeuralNetwork = FeedForward(NeuralNetwork)
        NeuralNetwork = BackPropogation(NeuralNetwork, ActivationFunction)

        if LossFunction == "MeanSquareError"
            loss[i] = MeanSquareError(NeuralNetwork)
        elseif LossFunction == "SumOfSquaresError"
            loss[i] = SumOfSquaresError(NeuralNetwork)
        elseif LossFunction == "CrossEntropy"
            loss[i] = CrossEntropy(NeuralNetwork)
        end
        print(string("Epoch: ", i, " of ", Epochs, ". Loss: ", loss[i]))

        if i >= 0.1 * Epochs
            if EarlyStopping == true
                if loss[i] >= loss[i-1]
                    BadCounter = BadCounter + 1
                    if BadCounter == Patience
                        break
                    end
                else
                    BadCounter = 0
                end
            else
            end
        end
    end

    return NeuralNetwork, loss
end