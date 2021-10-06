module ScratchNeuralNetwork

using LinearAlgebra, Plots

include("ActivationFunctions/Sigmoid.jl")
include("ActivationFunctions/Tanh.jl")
include("ActivationFunctions/ReLU.jl")
include("ActivationFunctionDerivatives/SigmoidDerivative.jl")
include("ActivationFunctionDerivatives/TanhDerivative.jl")
include("ActivationFunctionDerivatives/ReLUDerivative.jl")
include("LossFunctions/SumOfSquaresError.jl")
include("LossFunctions/MeanSquareError.jl")
include("LossFunctions/CrossEntropy.jl")
include("Architecture/FeedForward.jl")
include("Architecture/BackPropagation.jl")
include("Architecture/RunNeuralNetwork.jl")
include("Architecture/NeuralNetwork.jl")
#include("Plotting/PlotLossFunction.jl")

# Exports

export Sigmoid
export Tanh
export ReLU
export SigmoidDerivative
export TanhDerivative
export ReLUDerivative
export SumOfSquaresError
export MeanSquareError
export CrossEntropy
export FeedForward
export BackPropagation
export RunNeuralNetwork
export NeuralNetwork
#export PlotLossFunction

end
