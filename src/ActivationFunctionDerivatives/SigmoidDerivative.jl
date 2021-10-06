"""
    SigmoidDerivative(x)

Derivative of the Sigmoid activation function.

    Usage:
```julia-repl
SigmoidDerivative(x)
```
Arguments:
- `x` : Input value.
"""
function SigmoidDerivative(x::Float64)
    x₁ = x * (1.0 - x)
    return x₁
end