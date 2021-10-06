"""
    ReLUDerivative(x)

Derivative of the rectified linear unit activation function.

    Usage:
```julia-repl
ReLUDerivative(x)
```
Arguments:
- `x` : Input value.
"""
function ReLUDerivative(x::Float64)
    if x < 0
        x₁ = 0
    elseif x == 0
        x₁ = Inf
    elseif x > 0
        x₁ = 1
    end
    return x₁
end