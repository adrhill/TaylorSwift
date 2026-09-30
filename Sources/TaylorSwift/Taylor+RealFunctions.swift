import RealModule

// Functions beyond `ElementaryFunctions`, mirroring `RealFunctions` of the scalar type.
// `Taylor` does not conform to `RealFunctions` itself, because the derivatives of the
// gamma function are not available.

extension Taylor {
    /// The series of the path derivative `x'(t)`, with the coefficients `(k + 1) c_{k+1}`.
    ///
    /// Differentiation loses one order. The result is padded with a trailing zero, so its
    /// last coefficient is not meaningful.
    @inlinable
    internal var pathDerivative: Taylor {
        var result = [Scalar](repeating: .zero, count: storage.count)
        for k in 1..<storage.count {
            result[k - 1] = Scalar(k) * storage[k]
        }
        return Taylor(unchecked: result)
    }

    /// `2^x`.
    @inlinable
    public static func exp2(_ x: Taylor) -> Taylor {
        var result = exp(x * Scalar.log(2))
        result.storage[0] = Scalar.exp2(x.value)
        return result
    }

    /// `10^x`.
    @inlinable
    public static func exp10(_ x: Taylor) -> Taylor {
        var result = exp(x * Scalar.log(10))
        result.storage[0] = Scalar.exp10(x.value)
        return result
    }

    /// The base-2 logarithm.
    @inlinable
    public static func log2(_ x: Taylor) -> Taylor {
        var result = log(x) / Scalar.log(2)
        result.storage[0] = Scalar.log2(x.value)
        return result
    }

    /// The base-10 logarithm.
    @inlinable
    public static func log10(_ x: Taylor) -> Taylor {
        var result = log(x) / Scalar.log(10)
        result.storage[0] = Scalar.log10(x.value)
        return result
    }

    /// The two-argument inverse tangent: the angle of the point `(x, y)`.
    @inlinable
    public static func atan2(y: Taylor, x: Taylor) -> Taylor {
        let n = commonCount(y, x)
        let value = Scalar.atan2(y: y.value, x: x.value)
        let y = Taylor(unchecked: y.coefficients(count: n))
        let x = Taylor(unchecked: x.coefficients(count: n))
        if y.hasZeroHigherCoefficients && x.hasZeroHigherCoefficients {
            return x.constantPath(value)
        }
        // v' = (x y' - y x') / (x² + y²)
        let g = (x * y.pathDerivative - y * x.pathDerivative) / (x * x + y * y)
        var v = [Scalar](repeating: .zero, count: n)
        v[0] = value
        for k in 1..<n {
            v[k] = g.storage[k - 1] / Scalar(k)
        }
        return Taylor(unchecked: v)
    }

    /// The Euclidean norm `√(x² + y²)` of the point `(x, y)`.
    ///
    /// Like `Scalar.hypot`, this avoids overflow and underflow in the squares, also in the
    /// higher coefficients. The norm is not differentiable at the origin. There, with
    /// `x = t^m ξ` and `y = t^m η`, where `ξ₀` or `η₀` is nonzero, the result is the
    /// expansion `t^m hypot(ξ, η)` for `t → 0⁺`, as for ``abs(_:)``.
    @inlinable
    public static func hypot(_ x: Taylor, _ y: Taylor) -> Taylor {
        let n = commonCount(x, y)
        let value = Scalar.hypot(x.value, y.value)
        let (x, y) = (x.coefficients(count: n), y.coefficients(count: n))
        guard let m = x.indices.first(where: { k in k > 0 && (x[k] != .zero || y[k] != .zero) })
        else {
            return Taylor(unchecked: x).constantPath(value)
        }
        if value == .zero {
            let shifted = hypot(
                Taylor(unchecked: Array(x[m...])), Taylor(unchecked: Array(y[m...])))
            return Taylor(unchecked: [Scalar](repeating: .zero, count: m) + shifted.storage)
        }
        guard value.isFinite && x.allSatisfy(\.isFinite) && y.allSatisfy(\.isFinite) else {
            let (x, y) = (Taylor(unchecked: x), Taylor(unchecked: y))
            var result = sqrt(x * x + y * y)
            result.storage[0] = value
            return result
        }

        // Differentiating v² = x² + y² gives, with the direction cosines a = x₀ / v₀ and
        // b = y₀ / v₀,
        //
        //     v_k = a x_k + b y_k + (1 / 2) Σ_{j=1}^{k-1} (x_j x_{k-j} + y_j y_{k-j} - v_j v_{k-j}) / v₀
        //
        // Unlike the squares in √(x² + y²), none of these terms overflows or underflows
        // unless its value does, provided that each product is divided by v₀ through its
        // larger factor.
        func quotient(_ p: Scalar, _ q: Scalar) -> Scalar {
            p.magnitude >= q.magnitude ? (p / value) * q : (q / value) * p
        }
        let (a, b) = (x[0] / value, y[0] / value)
        var v = [Scalar](repeating: .zero, count: n)
        v[0] = value
        for k in 1..<n {
            var sum = Scalar.zero
            for j in 1..<k {
                sum += quotient(x[j], x[k - j]) + quotient(y[j], y[k - j])
                sum -= quotient(v[j], v[k - j])
            }
            v[k] = a * x[k] + b * y[k] + sum / 2
        }
        return Taylor(unchecked: v)
    }

    /// The error function.
    @inlinable
    public static func erf(_ x: Taylor) -> Taylor {
        let value = Scalar.erf(x.value)
        if x.hasZeroHigherCoefficients {
            return x.constantPath(value)
        }
        // v' = u' (2 / √π) exp(-u²)
        let g = exp(-(x * x)) * (2 / Scalar.sqrt(.pi))
        return integral(of: g.storage, along: x.storage, value: value)
    }

    /// The complementary error function.
    @inlinable
    public static func erfc(_ x: Taylor) -> Taylor {
        let value = Scalar.erfc(x.value)
        if x.hasZeroHigherCoefficients {
            return x.constantPath(value)
        }
        // v' = -u' (2 / √π) exp(-u²)
        let g = exp(-(x * x)) * (-2 / Scalar.sqrt(.pi))
        return integral(of: g.storage, along: x.storage, value: value)
    }

    /// The absolute value.
    ///
    /// At a zero primal value, the sign of `x(t)` for small `t > 0` is that of its first
    /// nonzero coefficient, and the result is the expansion for `t → 0⁺`. It is exact when
    /// `|x(t)|` is smooth, i.e. when that coefficient has an even index, as for `|-t²| = t²`.
    /// Otherwise it is one-sided, as for `|t|`, which gets the derivative `1`.
    @inlinable
    public static func abs(_ x: Taylor) -> Taylor {
        let leading = x.storage.first { $0 != .zero } ?? x.value
        return leading.sign == .minus ? -x : x
    }
}
