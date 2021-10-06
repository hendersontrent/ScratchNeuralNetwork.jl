"""
    Tanh(x)

Hyperbolic tangent activation function.

    Usage:
```julia-repl
Tanh(x)
```
Arguments:
- `x` : Input value.
"""
function Tanh(x::Float64)
    x₁ = tanh(x)
    return x₁
end