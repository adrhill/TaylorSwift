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
    /// The norm is not differentiable at the origin: if both primal values are zero, the
    /// higher coefficients are infinite or NaN, unless `x` and `y` are constant paths.
    @inlinable
    public static func hypot(_ x: Taylor, _ y: Taylor) -> Taylor {
        var result = sqrt(x * x + y * y)
        result.storage[0] = Scalar.hypot(x.value, y.value)
        return result
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
    /// The absolute value is not differentiable at zero. There, this function acts as the
    /// identity (or as the negation for a negative zero), which picks the derivative `±1`.
    @inlinable
    public static func abs(_ x: Taylor) -> Taylor {
        x.value.sign == .minus ? -x : x
    }
}
