module ScratchNeuralNetwork

using LinearAlgebra, Plots

include("ActivationFunctions/Sigmoid.jl")
include("ActivationFunctionDerivatives/SigmoidDerivative.jl")
include("LossFunctions/SumOfSquaresError.jl")
include("Architecture/FeedForward.jl")
include("Architecture/BackPropagation.jl")
include("Architecture/RunNeuralNetwork.jl")
#include("Plotting/PlotLossFunction.jl")

# Exports

export Sigmoid
export SigmoidDerivative
export SumOfSquaresError
export FeedForward
export BackPropagation
export RunNeuralNetwork
#export PlotLossFunction

end
