"""
    Sigmoid(x)

Sigmoid activation function.

    Usage:
```julia-repl
Sigmoid(x)
```
Arguments:
- `x` : Input value.
"""
function Sigmoid(x::Float64)
    x₁ = 1.0 / (1.0 + exp(-x))
    return x₁
end