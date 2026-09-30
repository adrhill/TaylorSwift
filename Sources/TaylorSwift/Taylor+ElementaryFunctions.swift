import RealModule

// The recurrences below follow from differentiating a defining relation of each function
// and comparing coefficients. With `u(t) = Σ u_k t^k`, the derivative has the coefficients
// `(u')_k = (k + 1) u_{k+1}`, so that e.g. `v = exp(u)` with `v' = u' v` gives
//
//     k v_k = Σ_{j=1}^{k} j u_j v_{k-j}.
//
// See Griewank & Walther, "Evaluating Derivatives" (2nd ed.), chapter 13.

extension Taylor {
    /// Solves `v' = u' g` for the coefficients of `v`, given `v_0 = value`:
    ///
    ///     v_k = (1 / k) Σ_{j=1}^{k} j u_j g_{k-j}
    ///
    /// This computes `v = φ(u)` from the series `g = φ'(u)` of the derivative.
    @inlinable
    internal static func integral(
        of g: [Scalar], along u: [Scalar], value: Scalar
    ) -> Taylor {
        let n = u.count
        var v = [Scalar](repeating: .zero, count: n)
        v[0] = value
        for k in 1..<n {
            var sum = Scalar.zero
            for j in 1...k {
                sum += Scalar(j) * u[j] * g[k - j]
            }
            v[k] = sum / Scalar(k)
        }
        return Taylor(unchecked: v)
    }

    /// Solves `w v' = w'` for the coefficients of `v = log(w)`, given `v_0 = value`:
    ///
    ///     v_k = (w_k - (1 / k) Σ_{j=1}^{k-1} j v_j w_{k-j}) / w_0
    @inlinable
    internal static func logarithm(of w: [Scalar], value: Scalar) -> Taylor {
        let n = w.count
        var v = [Scalar](repeating: .zero, count: n)
        v[0] = value
        for k in 1..<n {
            var sum = Scalar.zero
            for j in 1..<k {
                sum += Scalar(j) * v[j] * w[k - j]
            }
            v[k] = (w[k] - sum / Scalar(k)) / w[0]
        }
        return Taylor(unchecked: v)
    }

    /// Solves `u v' = r u' v` for the coefficients of `v = u^r`, given `v_0 = value`:
    ///
    ///     v_k = (1 / (k u_0)) Σ_{j=1}^{k} (r j - (k - j)) u_j v_{k-j}
    ///
    /// The closure returns the weight `r j - (k - j)` for the arguments `(j, k - j)`.
    @inlinable
    internal static func power(
        of u: [Scalar], value: Scalar, weight: (Scalar, Scalar) -> Scalar
    ) -> Taylor {
        let n = u.count
        var v = [Scalar](repeating: .zero, count: n)
        v[0] = value
        for k in 1..<n {
            var sum = Scalar.zero
            for j in 1...k {
                sum += weight(Scalar(j), Scalar(k - j)) * u[j] * v[k - j]
            }
            v[k] = sum / (Scalar(k) * u[0])
        }
        return Taylor(unchecked: v)
    }

    /// The coefficients of `sin(u)` and `cos(u)`, or of `sinh(u)` and `cosh(u)`.
    ///
    /// Solves the coupled equations `s' = u' c` and `c' = ∓u' s`, where the sign is
    /// negative for the circular and positive for the hyperbolic functions.
    @inlinable
    internal static func sineAndCosine(
        of u: [Scalar], values: (sine: Scalar, cosine: Scalar), hyperbolic: Bool
    ) -> (sine: Taylor, cosine: Taylor) {
        let n = u.count
        var s = [Scalar](repeating: .zero, count: n)
        var c = [Scalar](repeating: .zero, count: n)
        s[0] = values.sine
        c[0] = values.cosine
        for k in 1..<n {
            var sineSum = Scalar.zero
            var cosineSum = Scalar.zero
            for j in 1...k {
                let weight = Scalar(j) * u[j]
                sineSum += weight * c[k - j]
                cosineSum += weight * s[k - j]
            }
            s[k] = sineSum / Scalar(k)
            c[k] = (hyperbolic ? cosineSum : -cosineSum) / Scalar(k)
        }
        return (Taylor(unchecked: s), Taylor(unchecked: c))
    }

    /// The coefficients of `v = tan(u)` or `v = tanh(u)`, given `v_0 = value`.
    ///
    /// Solves `v' = u' w` alongside `w = 1 ± v²`, where the sign is positive for the
    /// circular and negative for the hyperbolic tangent. `derivative` is `w_0`.
    @inlinable
    internal static func tangent(
        of u: [Scalar], value: Scalar, derivative: Scalar, hyperbolic: Bool
    ) -> Taylor {
        let n = u.count
        var v = [Scalar](repeating: .zero, count: n)
        var w = [Scalar](repeating: .zero, count: n)
        v[0] = value
        w[0] = derivative
        for k in 1..<n {
            var sum = Scalar.zero
            for j in 1...k {
                sum += Scalar(j) * u[j] * w[k - j]
            }
            v[k] = sum / Scalar(k)

            var square = Scalar.zero
            for j in 0...k {
                square += v[j] * v[k - j]
            }
            w[k] = hyperbolic ? -square : square
        }
        return Taylor(unchecked: v)
    }

    /// The power `v = u^r` with a positive exponent `r = numerator / denominator` at a zero
    /// primal value, for a series `u` that is not a constant path.
    ///
    /// With `u = t^m w`, where `w₀` is the first nonzero coefficient, the power is
    /// `v = t^p w^r` with `p = m r` for `t → 0⁺`. Like ``abs(_:)`` at zero, this is the
    /// exact expansion if `v` is smooth at zero, and the one-sided expansion otherwise:
    ///
    /// - The coefficients below the order `p` vanish.
    /// - If `p` is whole, the coefficients from `p` on are those of `w^r`. Those beyond
    ///   the order `N - m + p` depend on coefficients of `u` beyond the truncation order,
    ///   and are NaN.
    /// - Otherwise the derivatives of the orders `k > p` diverge as `t → 0⁺`. Their
    ///   coefficients are infinite, with the sign of `p (p - 1) ⋯ (p - k + 1) w₀^r`.
    ///
    /// If `w₀^r` is NaN, as for a negative `w₀` and a fractional `r`, then so are all
    /// coefficients but the value.
    @inlinable
    internal static func powerAtZero(
        _ u: Taylor, numerator: Scalar, denominator: Scalar, value: Scalar,
        power: (Taylor) -> Taylor
    ) -> Taylor {
        let n = u.storage.count
        let m = u.storage.firstIndex { $0 != .zero }!
        let p = Scalar(m) * numerator / denominator
        let w = power(Taylor(unchecked: Array(u.storage[m...])))
        // The order `p` as an integer, if it is whole and within the truncation order.
        let wholeOrder = (0..<n).first { Scalar($0) == p }
        var v = [Scalar](repeating: .zero, count: n)
        v[0] = value
        for k in 1..<n {
            if w.value.isNaN {
                v[k] = .nan
            } else if Scalar(k) < p {
                continue
            } else if let wholeOrder {
                let j = k - wholeOrder
                v[k] = j < w.storage.count ? w.storage[j] : .nan
            } else {
                // The factors p - i are negative for i > p.
                let negativeFactors = (0..<k).count { Scalar($0) > p }
                let positive = (negativeFactors % 2 == 0) == (w.value.sign == .plus)
                v[k] = positive ? .infinity : -.infinity
            }
        }
        return Taylor(unchecked: v)
    }

    /// The power `x^n` by repeated squaring, which is exact for polynomials and
    /// well-defined for a zero primal value.
    @inlinable
    internal static func integerPower(_ x: Taylor, _ n: UInt) -> Taylor {
        var result = x.constantPath(1)
        var base = x
        var n = n
        while n > 0 {
            if n & 1 == 1 {
                result *= base
            }
            n >>= 1
            if n > 0 {
                base *= base
            }
        }
        return result
    }
}

// MARK: - ElementaryFunctions

extension Taylor: ElementaryFunctions {
    /// The exponential function `e^x`.
    @inlinable
    public static func exp(_ x: Taylor) -> Taylor {
        let value = Scalar.exp(x.value)
        if x.hasZeroHigherCoefficients {
            return x.constantPath(value)
        }
        // v' = u' v
        let u = x.storage
        var v = [Scalar](repeating: .zero, count: u.count)
        v[0] = value
        for k in 1..<u.count {
            var sum = Scalar.zero
            for j in 1...k {
                sum += Scalar(j) * u[j] * v[k - j]
            }
            v[k] = sum / Scalar(k)
        }
        return Taylor(unchecked: v)
    }

    /// `e^x - 1`, accurate for `x` close to zero.
    @inlinable
    public static func expMinusOne(_ x: Taylor) -> Taylor {
        var result = exp(x)
        result.storage[0] = Scalar.expMinusOne(x.value)
        return result
    }

    /// The natural logarithm.
    @inlinable
    public static func log(_ x: Taylor) -> Taylor {
        let value = Scalar.log(x.value)
        if x.hasZeroHigherCoefficients {
            return x.constantPath(value)
        }
        return logarithm(of: x.storage, value: value)
    }

    /// `log(1 + x)`, accurate for `x` close to zero.
    @inlinable
    public static func log(onePlus x: Taylor) -> Taylor {
        let value = Scalar.log(onePlus: x.value)
        if x.hasZeroHigherCoefficients {
            return x.constantPath(value)
        }
        return logarithm(of: (1 + x).storage, value: value)
    }

    /// The sine.
    @inlinable
    public static func sin(_ x: Taylor) -> Taylor {
        let values = (sine: Scalar.sin(x.value), cosine: Scalar.cos(x.value))
        if x.hasZeroHigherCoefficients {
            return x.constantPath(values.sine)
        }
        return sineAndCosine(of: x.storage, values: values, hyperbolic: false).sine
    }

    /// The cosine.
    @inlinable
    public static func cos(_ x: Taylor) -> Taylor {
        let values = (sine: Scalar.sin(x.value), cosine: Scalar.cos(x.value))
        if x.hasZeroHigherCoefficients {
            return x.constantPath(values.cosine)
        }
        return sineAndCosine(of: x.storage, values: values, hyperbolic: false).cosine
    }

    /// The tangent.
    @inlinable
    public static func tan(_ x: Taylor) -> Taylor {
        let value = Scalar.tan(x.value)
        if x.hasZeroHigherCoefficients {
            return x.constantPath(value)
        }
        // v' = u' (1 + v²)
        return tangent(
            of: x.storage, value: value, derivative: 1 + value * value, hyperbolic: false)
    }

    /// The hyperbolic sine.
    @inlinable
    public static func sinh(_ x: Taylor) -> Taylor {
        let values = (sine: Scalar.sinh(x.value), cosine: Scalar.cosh(x.value))
        if x.hasZeroHigherCoefficients {
            return x.constantPath(values.sine)
        }
        return sineAndCosine(of: x.storage, values: values, hyperbolic: true).sine
    }

    /// The hyperbolic cosine.
    @inlinable
    public static func cosh(_ x: Taylor) -> Taylor {
        let values = (sine: Scalar.sinh(x.value), cosine: Scalar.cosh(x.value))
        if x.hasZeroHigherCoefficients {
            return x.constantPath(values.cosine)
        }
        return sineAndCosine(of: x.storage, values: values, hyperbolic: true).cosine
    }

    /// The hyperbolic tangent.
    @inlinable
    public static func tanh(_ x: Taylor) -> Taylor {
        let value = Scalar.tanh(x.value)
        if x.hasZeroHigherCoefficients {
            return x.constantPath(value)
        }
        // v' = u' (1 - v²), where 1 - v₀² = 1 / cosh²(u₀) is evaluated without cancellation.
        let cosh = Scalar.cosh(x.value)
        return tangent(
            of: x.storage, value: value, derivative: 1 / (cosh * cosh), hyperbolic: true)
    }

    /// The inverse sine.
    @inlinable
    public static func asin(_ x: Taylor) -> Taylor {
        let value = Scalar.asin(x.value)
        if x.hasZeroHigherCoefficients {
            return x.constantPath(value)
        }
        // v' = u' / √(1 - u²)
        let g = 1 / sqrt((1 - x) * (1 + x))
        return integral(of: g.storage, along: x.storage, value: value)
    }

    /// The inverse cosine.
    @inlinable
    public static func acos(_ x: Taylor) -> Taylor {
        let value = Scalar.acos(x.value)
        if x.hasZeroHigherCoefficients {
            return x.constantPath(value)
        }
        // v' = -u' / √(1 - u²)
        let g = -1 / sqrt((1 - x) * (1 + x))
        return integral(of: g.storage, along: x.storage, value: value)
    }

    /// The inverse tangent.
    @inlinable
    public static func atan(_ x: Taylor) -> Taylor {
        let value = Scalar.atan(x.value)
        if x.hasZeroHigherCoefficients {
            return x.constantPath(value)
        }
        // v' = u' / (1 + u²)
        let g = 1 / (1 + x * x)
        return integral(of: g.storage, along: x.storage, value: value)
    }

    /// The inverse hyperbolic sine.
    @inlinable
    public static func asinh(_ x: Taylor) -> Taylor {
        let value = Scalar.asinh(x.value)
        if x.hasZeroHigherCoefficients {
            return x.constantPath(value)
        }
        // v' = u' / √(1 + u²)
        let g = 1 / sqrt(1 + x * x)
        return integral(of: g.storage, along: x.storage, value: value)
    }

    /// The inverse hyperbolic cosine.
    @inlinable
    public static func acosh(_ x: Taylor) -> Taylor {
        let value = Scalar.acosh(x.value)
        if x.hasZeroHigherCoefficients {
            return x.constantPath(value)
        }
        // v' = u' / √(u² - 1)
        let g = 1 / sqrt((x - 1) * (x + 1))
        return integral(of: g.storage, along: x.storage, value: value)
    }

    /// The inverse hyperbolic tangent.
    @inlinable
    public static func atanh(_ x: Taylor) -> Taylor {
        let value = Scalar.atanh(x.value)
        if x.hasZeroHigherCoefficients {
            return x.constantPath(value)
        }
        // v' = u' / (1 - u²)
        let g = 1 / ((1 - x) * (1 + x))
        return integral(of: g.storage, along: x.storage, value: value)
    }

    /// The square root.
    ///
    /// At a zero primal value, the result is the expansion for `t → 0⁺`: with `x = t^m w`,
    /// it is `t^(m/2) √w`. Its coefficients are finite up to the order that the truncated
    /// `x` determines if `m` is even, and infinite from the order `m / 2` on if `m` is odd.
    @inlinable
    public static func sqrt(_ x: Taylor) -> Taylor {
        let value = Scalar.sqrt(x.value)
        if x.hasZeroHigherCoefficients {
            return x.constantPath(value)
        }
        if x.value == .zero {
            return powerAtZero(x, numerator: 1, denominator: 2, value: value) { sqrt($0) }
        }
        // v² = u, so that 2 v₀ v_k = u_k - Σ_{j=1}^{k-1} v_j v_{k-j}
        let u = x.storage
        var v = [Scalar](repeating: .zero, count: u.count)
        v[0] = value
        for k in 1..<u.count {
            var sum = u[k]
            for j in 1..<k {
                sum -= v[j] * v[k - j]
            }
            v[k] = sum / (2 * value)
        }
        return Taylor(unchecked: v)
    }

    /// The real `n`-th root.
    ///
    /// For odd `n`, the root of a negative value is the negative real root. At a zero primal
    /// value and for positive `n`, the result is the expansion for `t → 0⁺`, as for
    /// ``sqrt(_:)``.
    @inlinable
    public static func root(_ x: Taylor, _ n: Int) -> Taylor {
        if n == 1 {
            return x
        }
        let value = Scalar.root(x.value, n)
        if x.hasZeroHigherCoefficients {
            return x.constantPath(value)
        }
        if x.value == .zero && n > 0 {
            return powerAtZero(x, numerator: 1, denominator: Scalar(n), value: value) {
                root($0, n)
            }
        }
        // The weight is r j - (k - j) for r = 1 / n, written to be exact in the numerator.
        let degree = Scalar(n)
        return power(of: x.storage, value: value) { j, rest in (j - degree * rest) / degree }
    }

    /// The power `x^n` with an integer exponent.
    ///
    /// Unlike the other overloads of `pow`, this is well-defined for a zero or negative
    /// primal value.
    @inlinable
    public static func pow(_ x: Taylor, _ n: Int) -> Taylor {
        let value = Scalar.pow(x.value, n)
        if x.hasZeroHigherCoefficients {
            return x.constantPath(value)
        }
        var result = integerPower(x, n.magnitude)
        if n < 0 {
            result = 1 / result
        }
        result.storage[0] = value
        return result
    }

    /// The power `x^y` with a scalar exponent.
    ///
    /// Like `Scalar.pow` with a real exponent, this is `exp(y log(x))` and hence NaN for a
    /// negative primal value, even if `y` is a whole number; use the overload with an
    /// integer exponent for those.
    ///
    /// At a zero primal value, a non-negative whole `y` gives the exact power. A positive
    /// fractional `y` gives the expansion for `t → 0⁺`: with `x = t^m w`, it is
    /// `t^(m y) w^y`, whose coefficients vanish below the order `m y` and are infinite
    /// beyond it, unless `m y` is whole. A negative `y` gives infinite or NaN coefficients.
    @inlinable @_disfavoredOverload
    public static func pow(_ x: Taylor, _ y: Scalar) -> Taylor {
        let value = Scalar.pow(x.value, y)
        if x.hasZeroHigherCoefficients {
            return x.constantPath(value)
        }
        if x.value == .zero && y >= 0 && y == y.rounded(.towardZero) && y.isFinite {
            // Repeated squaring, driven by the binary digits of the integer-valued `y`.
            var result = x.constantPath(1)
            var base = x
            var exponent = y
            while exponent > 0 {
                let half = (exponent / 2).rounded(.towardZero)
                if exponent - 2 * half == 1 {
                    result *= base
                }
                exponent = half
                if exponent > 0 {
                    base *= base
                }
            }
            result.storage[0] = value
            return result
        }
        if x.value == .zero && y > 0 {
            return powerAtZero(x, numerator: y, denominator: 1, value: value) { pow($0, y) }
        }
        return power(of: x.storage, value: value) { j, rest in y * j - rest }
    }

    /// The power `x^y`.
    ///
    /// If `y` is a constant path, this is the same as `pow` with a scalar exponent.
    /// Otherwise the power is `exp(y log(x))`, which requires a positive primal value
    /// of `x`.
    @inlinable
    public static func pow(_ x: Taylor, _ y: Taylor) -> Taylor {
        let n = commonCount(x, y)
        let base = Taylor(unchecked: x.coefficients(count: n))
        if y.hasZeroHigherCoefficients {
            return pow(base, y.value)
        }
        let value = Scalar.pow(x.value, y.value)
        if base == .zero && y.value > 0 {
            // 0^y is identically zero for positive exponents.
            return base.constantPath(value)
        }
        var result = exp(y * log(base))
        result.storage[0] = value
        return result
    }
}
