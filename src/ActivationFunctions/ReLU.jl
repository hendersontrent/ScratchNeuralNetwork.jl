"""
    ReLU(x)

Rectified linear unit activation function.

    Usage:
```julia-repl
ReLU(x)
```
Arguments:
- `x` : Input value.
"""
function ReLU(x::Float64)
    x₁ = max(0.0, x)
    return x₁
end