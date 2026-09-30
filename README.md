# TaylorSwift

Taylor-mode automatic differentiation for Swift: higher-order derivatives in a single
forward pass.


> [!WARNING]  
> Don't use this. This package was vibe-coded as a joke on a train ride back from [EuroAD29](https://cambridge-iccs.github.io/euroad29/).

A `Taylor` value is a truncated Taylor series
`c₀ + c₁ t + … + c_N t^N` with normalized coefficients `c_k = x⁽ᵏ⁾(0) / k!`.
Arithmetic and elementary functions propagate all coefficients at once, so evaluating `f`
on the path `x₀ + t` yields `f(x₀), f'(x₀), …, f⁽ᴺ⁾(x₀)` in `O(N²)` time per nonlinear
operation, instead of the exponential cost of nesting first-order differentiation.

## Usage

```swift
import TaylorSwift

// All derivatives up to the fourth order, in one pass.
derivatives(of: { sin($0) * exp($0) }, at: 0.0, through: 4)  // [0, 1, 2, 2, 0]

// A single derivative of a given order.
derivative(of: { $0 * $0 * $0 }, at: 2.0, order: 2)  // 12

// The normalized Taylor coefficients f⁽ᵏ⁾(x) / k!.
taylorCoefficients(of: { 1 / (1 - $0) }, at: 0.0, order: 3)  // [1, 1, 1, 1]
```

The number type can also be used directly:

```swift
let x = Taylor.variable(2.0, order: 3)  // the path 2 + t
let y = x * x * x

y.coefficients  // [8, 12, 6, 1]
y.derivatives   // [8, 12, 12, 6]
y.derivative(2) // 12
```

`Taylor` is generic over the scalar type (`Float`, `Double`, … — any
[`Real`](https://github.com/apple/swift-numerics) type) and conforms to
`ElementaryFunctions` and `AlgebraicField` from swift-numerics, so that generic numerical
code is differentiable as is:

```swift
import RealModule

func logistic<T: ElementaryFunctions & AlgebraicField>(_ x: T) -> T {
    1 / (1 + T.exp(-x))
}

derivatives(of: logistic, at: 0.0, through: 3)  // [0.5, 0.25, 0, -0.125]
```

### Several variables

Taylor mode differentiates along one direction at a time. Functions of several variables
take an array of `Taylor` values:

```swift
func f(_ v: [Taylor<Double>]) -> Taylor<Double> {
    v[0] * v[0] * v[1] + sin(v[1])
}
let x = [1.0, 2.0]

// k-th derivative of t ↦ f(x + t v)
directionalDerivative(of: f, at: x, along: [1, 0], order: 2)  // 4

gradient(of: f, at: x)   // [4, 1 + cos(2)]
hessian(of: f, at: x)    // [[4, 2], [2, -sin(2)]]
laplacian(of: f, at: x)  // 4 - sin(2)

// ∂³f / ∂x₀² ∂x₁
mixedPartial(of: f, at: x, withRespectTo: [0, 0, 1])  // 2
```

Mixed partial derivatives are recovered exactly from directional derivatives through
polarization identities.

## Supported operations

- `+`, `-`, `*`, `/` between series, and between series and scalars
- `exp`, `exp2`, `exp10`, `expm1`, `log`, `log2`, `log10`, `log1p`
- `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`
- `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
- `sqrt`, `cbrt`, `root`, `pow` (integer, scalar, and series exponents), `hypot`
- `erf`, `erfc`, `abs`
- `<`, `<=`, `>`, `>=` on the primal value, for control flow

## Semantics

- **Orders.** All series in a computation must have the same order; mixing orders is a
  precondition failure. Series with a single coefficient are exact constants, which
  combine with series of any order. Scalars and literals are lifted to such constants.
- **Literals.** Swift reads `Taylor(3.0)` as a literal of type `Taylor`, whose scalar type
  it cannot infer. Write `Taylor<Double>(3.0)` instead.
- **Singularities.** At points where a function is not differentiable, such as `sqrt` at
  zero, the higher coefficients are infinite or NaN. `abs` at zero acts as the identity.
- **Powers.** `pow` with an integer exponent works for any base. As for scalars, `pow`
  with a real exponent is NaN for negative bases.

## Installation

Please don't.

## References

- A. Griewank and A. Walther, *Evaluating Derivatives*, 2nd edition, SIAM, 2008, chapter 13.
- J. Bettencourt, M. J. Johnson, and D. Duvenaud,
  *Taylor-Mode Automatic Differentiation for Higher-Order Derivatives in JAX*, 2019.
