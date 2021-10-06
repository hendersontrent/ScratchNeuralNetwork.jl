"""
    RunNeuralNetwork(epochs, NeuralNetwork)

Train a neural network.

    Usage:
```julia-repl
RunNeuralNetwork(epochs, NeuralNetwork)
```
Arguments:
- `epochs` : The number of iterations to train the network for.
- `NeuralNetwork` : The NeuralNetwork structure.
"""
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