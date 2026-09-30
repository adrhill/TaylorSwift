import RealModule

// Free functions named like their counterparts for `Double` in the C math library, so that
// the same function body can be evaluated on scalars and on `Taylor` series.

/// The exponential function `e^x`.
@inlinable
public func exp<Scalar: Real>(_ x: Taylor<Scalar>) -> Taylor<Scalar> { .exp(x) }

/// `2^x`.
@inlinable
public func exp2<Scalar: Real>(_ x: Taylor<Scalar>) -> Taylor<Scalar> { .exp2(x) }

/// `e^x - 1`, accurate for `x` close to zero.
@inlinable
public func expm1<Scalar: Real>(_ x: Taylor<Scalar>) -> Taylor<Scalar> { .expMinusOne(x) }

/// The natural logarithm.
@inlinable
public func log<Scalar: Real>(_ x: Taylor<Scalar>) -> Taylor<Scalar> { .log(x) }

/// The base-2 logarithm.
@inlinable
public func log2<Scalar: Real>(_ x: Taylor<Scalar>) -> Taylor<Scalar> { .log2(x) }

/// The base-10 logarithm.
@inlinable
public func log10<Scalar: Real>(_ x: Taylor<Scalar>) -> Taylor<Scalar> { .log10(x) }

/// `log(1 + x)`, accurate for `x` close to zero.
@inlinable
public func log1p<Scalar: Real>(_ x: Taylor<Scalar>) -> Taylor<Scalar> { .log(onePlus: x) }

/// The sine.
@inlinable
public func sin<Scalar: Real>(_ x: Taylor<Scalar>) -> Taylor<Scalar> { .sin(x) }

/// The cosine.
@inlinable
public func cos<Scalar: Real>(_ x: Taylor<Scalar>) -> Taylor<Scalar> { .cos(x) }

/// The tangent.
@inlinable
public func tan<Scalar: Real>(_ x: Taylor<Scalar>) -> Taylor<Scalar> { .tan(x) }

/// The inverse sine.
@inlinable
public func asin<Scalar: Real>(_ x: Taylor<Scalar>) -> Taylor<Scalar> { .asin(x) }

/// The inverse cosine.
@inlinable
public func acos<Scalar: Real>(_ x: Taylor<Scalar>) -> Taylor<Scalar> { .acos(x) }

/// The inverse tangent.
@inlinable
public func atan<Scalar: Real>(_ x: Taylor<Scalar>) -> Taylor<Scalar> { .atan(x) }

/// The two-argument inverse tangent: the angle of the point `(x, y)`.
@inlinable
public func atan2<Scalar: Real>(_ y: Taylor<Scalar>, _ x: Taylor<Scalar>) -> Taylor<Scalar> {
    .atan2(y: y, x: x)
}

/// The hyperbolic sine.
@inlinable
public func sinh<Scalar: Real>(_ x: Taylor<Scalar>) -> Taylor<Scalar> { .sinh(x) }

/// The hyperbolic cosine.
@inlinable
public func cosh<Scalar: Real>(_ x: Taylor<Scalar>) -> Taylor<Scalar> { .cosh(x) }

/// The hyperbolic tangent.
@inlinable
public func tanh<Scalar: Real>(_ x: Taylor<Scalar>) -> Taylor<Scalar> { .tanh(x) }

/// The inverse hyperbolic sine.
@inlinable
public func asinh<Scalar: Real>(_ x: Taylor<Scalar>) -> Taylor<Scalar> { .asinh(x) }

/// The inverse hyperbolic cosine.
@inlinable
public func acosh<Scalar: Real>(_ x: Taylor<Scalar>) -> Taylor<Scalar> { .acosh(x) }

/// The inverse hyperbolic tangent.
@inlinable
public func atanh<Scalar: Real>(_ x: Taylor<Scalar>) -> Taylor<Scalar> { .atanh(x) }

/// The square root.
@inlinable
public func sqrt<Scalar: Real>(_ x: Taylor<Scalar>) -> Taylor<Scalar> { .sqrt(x) }

/// The real cube root.
@inlinable
public func cbrt<Scalar: Real>(_ x: Taylor<Scalar>) -> Taylor<Scalar> { .root(x, 3) }

/// The power `x^n` with an integer exponent.
@inlinable
public func pow<Scalar: Real>(_ x: Taylor<Scalar>, _ n: Int) -> Taylor<Scalar> { .pow(x, n) }

/// The power `x^y` with a scalar exponent.
@inlinable @_disfavoredOverload
public func pow<Scalar: Real>(_ x: Taylor<Scalar>, _ y: Scalar) -> Taylor<Scalar> { .pow(x, y) }

/// The power `x^y`.
@inlinable
public func pow<Scalar: Real>(_ x: Taylor<Scalar>, _ y: Taylor<Scalar>) -> Taylor<Scalar> {
    .pow(x, y)
}

/// The Euclidean norm `√(x² + y²)` of the point `(x, y)`.
@inlinable
public func hypot<Scalar: Real>(_ x: Taylor<Scalar>, _ y: Taylor<Scalar>) -> Taylor<Scalar> {
    .hypot(x, y)
}

/// The error function.
@inlinable
public func erf<Scalar: Real>(_ x: Taylor<Scalar>) -> Taylor<Scalar> { .erf(x) }

/// The complementary error function.
@inlinable
public func erfc<Scalar: Real>(_ x: Taylor<Scalar>) -> Taylor<Scalar> { .erfc(x) }

/// The absolute value.
@inlinable
public func abs<Scalar: Real>(_ x: Taylor<Scalar>) -> Taylor<Scalar> { .abs(x) }
