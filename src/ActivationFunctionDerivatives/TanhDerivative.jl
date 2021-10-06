"""
    TanhDerivative(x)

Derivative of the hyperbolic tangent activation function.

    Usage:
```julia-repl
TanhDerivative(x)
```
Arguments:
- `x` : Input value.
"""
function TanhDerivative(x::Float64)
    x₁ = 1 - tanh2(x)
    return x₁
end