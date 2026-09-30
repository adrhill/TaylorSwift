import RealModule

// MARK: - Literals

extension Taylor: ExpressibleByIntegerLiteral {
    /// Creates a constant from an integer literal.
    @inlinable
    public init(integerLiteral value: Scalar.IntegerLiteralType) {
        self.init(Scalar(integerLiteral: value))
    }
}

extension Taylor: ExpressibleByFloatLiteral where Scalar: ExpressibleByFloatLiteral {
    /// Creates a constant from a floating-point literal.
    @inlinable
    public init(floatLiteral value: Scalar.FloatLiteralType) {
        self.init(Scalar(floatLiteral: value))
    }
}

// MARK: - Addition and subtraction

extension Taylor: AdditiveArithmetic {
    /// The constant zero.
    @inlinable
    public static var zero: Taylor { Taylor(.zero) }

    @inlinable
    public static func + (lhs: Taylor, rhs: Taylor) -> Taylor {
        if rhs.isConstant {
            return lhs + rhs.value
        }
        if lhs.isConstant {
            return lhs.value + rhs
        }
        _ = commonCount(lhs, rhs)
        var result = lhs.storage
        for k in result.indices {
            result[k] += rhs.storage[k]
        }
        return Taylor(unchecked: result)
    }

    @inlinable
    public static func - (lhs: Taylor, rhs: Taylor) -> Taylor {
        if rhs.isConstant {
            return lhs - rhs.value
        }
        if lhs.isConstant {
            return lhs.value - rhs
        }
        _ = commonCount(lhs, rhs)
        var result = lhs.storage
        for k in result.indices {
            result[k] -= rhs.storage[k]
        }
        return Taylor(unchecked: result)
    }

    @inlinable
    public static func += (lhs: inout Taylor, rhs: Taylor) {
        lhs = lhs + rhs
    }

    @inlinable
    public static func -= (lhs: inout Taylor, rhs: Taylor) {
        lhs = lhs - rhs
    }
}

// MARK: - Multiplication

extension Taylor: Numeric {
    /// The magnitude of the primal ``value``.
    @inlinable
    public var magnitude: Scalar { value.magnitude }

    /// Creates a constant from an integer, if it is exactly representable as a `Scalar`.
    @inlinable
    public init?<T: BinaryInteger>(exactly source: T) {
        guard let value = Scalar(exactly: source) else {
            return nil
        }
        self.init(value)
    }

    /// The Cauchy product `(u w)_k = Σ_{j=0}^{k} u_j w_{k-j}`.
    @inlinable
    public static func * (lhs: Taylor, rhs: Taylor) -> Taylor {
        if rhs.isConstant {
            return lhs * rhs.value
        }
        if lhs.isConstant {
            return lhs.value * rhs
        }
        let n = commonCount(lhs, rhs)
        let (u, w) = (lhs.storage, rhs.storage)
        var v = [Scalar](repeating: .zero, count: n)
        for k in 0..<n {
            var sum = Scalar.zero
            for j in 0...k {
                sum += u[j] * w[k - j]
            }
            v[k] = sum
        }
        return Taylor(unchecked: v)
    }

    @inlinable
    public static func *= (lhs: inout Taylor, rhs: Taylor) {
        lhs = lhs * rhs
    }
}

extension Taylor: SignedNumeric {
    @inlinable
    public static prefix func - (operand: Taylor) -> Taylor {
        Taylor(unchecked: operand.storage.map { -$0 })
    }

    @inlinable
    public mutating func negate() {
        self = -self
    }
}

// MARK: - Division

extension Taylor: AlgebraicField {
    /// The quotient `v = u / w`, from `v_k = (u_k - Σ_{j=0}^{k-1} v_j w_{k-j}) / w_0`.
    @inlinable
    public static func / (lhs: Taylor, rhs: Taylor) -> Taylor {
        if rhs.isConstant {
            return lhs / rhs.value
        }
        let n = commonCount(lhs, rhs)
        let (u, w) = (lhs.coefficients(count: n), rhs.storage)
        var v = [Scalar](repeating: .zero, count: n)
        for k in 0..<n {
            var sum = u[k]
            for j in 0..<k {
                sum -= v[j] * w[k - j]
            }
            v[k] = sum / w[0]
        }
        return Taylor(unchecked: v)
    }

    @inlinable
    public static func /= (lhs: inout Taylor, rhs: Taylor) {
        lhs = lhs / rhs
    }

    /// The multiplicative inverse, or `nil` if the primal ``value`` is zero or its
    /// reciprocal is not accurately representable (see `Scalar.reciprocal`).
    @inlinable
    public var reciprocal: Taylor? {
        value == .zero || value.reciprocal == nil ? nil : 1 / self
    }
}

// MARK: - Arithmetic with scalars

// A literal operand could become either a `Scalar` or a constant `Taylor`. Both give the
// same result, and `@_disfavoredOverload` resolves the ambiguity in favor of the latter.

extension Taylor {
    @inlinable @_disfavoredOverload
    public static func + (lhs: Taylor, rhs: Scalar) -> Taylor {
        var result = lhs
        result.storage[0] += rhs
        return result
    }

    @inlinable @_disfavoredOverload
    public static func + (lhs: Scalar, rhs: Taylor) -> Taylor {
        var result = rhs
        result.storage[0] = lhs + result.storage[0]
        return result
    }

    @inlinable @_disfavoredOverload
    public static func - (lhs: Taylor, rhs: Scalar) -> Taylor {
        var result = lhs
        result.storage[0] -= rhs
        return result
    }

    @inlinable @_disfavoredOverload
    public static func - (lhs: Scalar, rhs: Taylor) -> Taylor {
        var result = -rhs
        result.storage[0] = lhs - rhs.storage[0]
        return result
    }

    @inlinable @_disfavoredOverload
    public static func * (lhs: Taylor, rhs: Scalar) -> Taylor {
        Taylor(unchecked: lhs.storage.map { $0 * rhs })
    }

    @inlinable @_disfavoredOverload
    public static func * (lhs: Scalar, rhs: Taylor) -> Taylor {
        Taylor(unchecked: rhs.storage.map { lhs * $0 })
    }

    @inlinable @_disfavoredOverload
    public static func / (lhs: Taylor, rhs: Scalar) -> Taylor {
        Taylor(unchecked: lhs.storage.map { $0 / rhs })
    }

    @inlinable @_disfavoredOverload
    public static func / (lhs: Scalar, rhs: Taylor) -> Taylor {
        Taylor(lhs) / rhs
    }

    @inlinable @_disfavoredOverload
    public static func += (lhs: inout Taylor, rhs: Scalar) {
        lhs = lhs + rhs
    }

    @inlinable @_disfavoredOverload
    public static func -= (lhs: inout Taylor, rhs: Scalar) {
        lhs = lhs - rhs
    }

    @inlinable @_disfavoredOverload
    public static func *= (lhs: inout Taylor, rhs: Scalar) {
        lhs = lhs * rhs
    }

    @inlinable @_disfavoredOverload
    public static func /= (lhs: inout Taylor, rhs: Scalar) {
        lhs = lhs / rhs
    }
}

// MARK: - Comparison

// `Taylor` is deliberately not `Comparable`: ordering compares only the primal values,
// which would be inconsistent with `==` comparing all coefficients. The operators exist
// so that differentiated code can branch on the value, as in `x < 0 ? -x : x`.
extension Taylor {
    /// Compares the primal values.
    @inlinable
    public static func < (lhs: Taylor, rhs: Taylor) -> Bool { lhs.value < rhs.value }

    /// Compares the primal values.
    @inlinable
    public static func <= (lhs: Taylor, rhs: Taylor) -> Bool { lhs.value <= rhs.value }

    /// Compares the primal values.
    @inlinable
    public static func > (lhs: Taylor, rhs: Taylor) -> Bool { lhs.value > rhs.value }

    /// Compares the primal values.
    @inlinable
    public static func >= (lhs: Taylor, rhs: Taylor) -> Bool { lhs.value >= rhs.value }

    /// Compares the primal value to a scalar.
    @inlinable @_disfavoredOverload
    public static func < (lhs: Taylor, rhs: Scalar) -> Bool { lhs.value < rhs }

    /// Compares the primal value to a scalar.
    @inlinable @_disfavoredOverload
    public static func <= (lhs: Taylor, rhs: Scalar) -> Bool { lhs.value <= rhs }

    /// Compares the primal value to a scalar.
    @inlinable @_disfavoredOverload
    public static func > (lhs: Taylor, rhs: Scalar) -> Bool { lhs.value > rhs }

    /// Compares the primal value to a scalar.
    @inlinable @_disfavoredOverload
    public static func >= (lhs: Taylor, rhs: Scalar) -> Bool { lhs.value >= rhs }

    /// Compares a scalar to the primal value.
    @inlinable @_disfavoredOverload
    public static func < (lhs: Scalar, rhs: Taylor) -> Bool { lhs < rhs.value }

    /// Compares a scalar to the primal value.
    @inlinable @_disfavoredOverload
    public static func <= (lhs: Scalar, rhs: Taylor) -> Bool { lhs <= rhs.value }

    /// Compares a scalar to the primal value.
    @inlinable @_disfavoredOverload
    public static func > (lhs: Scalar, rhs: Taylor) -> Bool { lhs > rhs.value }

    /// Compares a scalar to the primal value.
    @inlinable @_disfavoredOverload
    public static func >= (lhs: Scalar, rhs: Taylor) -> Bool { lhs >= rhs.value }
}
