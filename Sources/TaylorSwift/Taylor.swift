import RealModule

/// A truncated Taylor series, the number type of Taylor-mode automatic differentiation.
///
/// A `Taylor` value represents a smooth path `x(t)` by its Taylor polynomial around `t = 0`:
///
///     x(t) = c₀ + c₁ t + c₂ t² + … + c_N t^N + O(t^(N+1))
///
/// The coefficients are *normalized*: `c_k = x⁽ᵏ⁾(0) / k!`. Every arithmetic operation and
/// elementary function on `Taylor` propagates all `N + 1` coefficients at once, so evaluating
/// a function `f` on the path `x₀ + t` yields `f(x₀)` and all derivatives `f'(x₀), …, f⁽ᴺ⁾(x₀)`
/// in a single pass, in `O(N²)` time per nonlinear operation. This avoids the exponential
/// blow-up of nesting first-order automatic differentiation.
///
///     let x = Taylor.variable(2.0, order: 3)
///     let y = x * x * x            // t ↦ (2 + t)³
///     y.derivatives                // [8.0, 12.0, 12.0, 6.0]
///
/// ### Orders and constants
///
/// All series taking part in a computation must have the same ``order``. The one exception
/// is a series with a single coefficient: it is an exact constant and combines with series
/// of any order. Scalars and numeric literals are lifted to such constants.
public struct Taylor<Scalar: Real> {
    /// The normalized coefficients; never empty.
    @usableFromInline
    internal var storage: [Scalar]

    @inlinable
    internal init(unchecked storage: [Scalar]) {
        assert(!storage.isEmpty, "A Taylor series needs at least one coefficient")
        self.storage = storage
    }

    /// Creates a constant, which combines with series of any order.
    @inlinable
    public init(_ value: Scalar) {
        self.init(unchecked: [value])
    }

    /// Creates a series from its normalized Taylor coefficients `c_k = x⁽ᵏ⁾(0) / k!`.
    ///
    /// - Precondition: `coefficients` is not empty.
    @inlinable
    public init(coefficients: [Scalar]) {
        precondition(!coefficients.isEmpty, "A Taylor series needs at least one coefficient")
        self.init(unchecked: coefficients)
    }

    /// Creates a series from the derivatives `x(0), x'(0), x''(0), …`.
    ///
    /// - Precondition: `derivatives` is not empty.
    @inlinable
    public init(derivatives: [Scalar]) {
        precondition(!derivatives.isEmpty, "A Taylor series needs at least one coefficient")
        var coefficients = derivatives
        var factorial: Scalar = 1
        for k in 1..<coefficients.count {
            factorial *= Scalar(k)
            coefficients[k] /= factorial
        }
        self.init(unchecked: coefficients)
    }

    /// The path `value + direction · t`, truncated at the given order.
    ///
    /// Evaluating a function on `variable(x₀, order: N)` computes its Taylor expansion
    /// around `x₀` up to order `N`.
    ///
    /// - Precondition: `order >= 0`.
    @inlinable
    public static func variable(_ value: Scalar, direction: Scalar = 1, order: Int) -> Taylor {
        precondition(order >= 0, "The order of a Taylor series must not be negative")
        var coefficients = [Scalar](repeating: .zero, count: order + 1)
        coefficients[0] = value
        if order >= 1 {
            coefficients[1] = direction
        }
        return Taylor(unchecked: coefficients)
    }
}

extension Taylor: Sendable where Scalar: Sendable {}

// MARK: - Accessors

extension Taylor {
    /// The normalized Taylor coefficients `c_k = x⁽ᵏ⁾(0) / k!`, for `k = 0, …, order`.
    @inlinable
    public var coefficients: [Scalar] { storage }

    /// The zeroth coefficient, i.e. the primal value `x(0)`.
    @inlinable
    public var value: Scalar { storage[0] }

    /// The truncation order `N`: the highest power of `t` that is tracked.
    @inlinable
    public var order: Int { storage.count - 1 }

    /// Whether this series is an exact constant that combines with series of any order.
    @inlinable
    public var isConstant: Bool { storage.count == 1 }

    /// The normalized Taylor coefficient `c_k = x⁽ᵏ⁾(0) / k!`.
    ///
    /// The higher coefficients of a constant are zero for every `k`.
    ///
    /// - Precondition: `0 <= k`, and `k <= order` unless the series is a constant.
    @inlinable
    public subscript(k: Int) -> Scalar {
        precondition(k >= 0, "The index of a Taylor coefficient must not be negative")
        if isConstant && k > 0 {
            return .zero
        }
        precondition(k <= order, "The Taylor coefficient is beyond the truncation order")
        return storage[k]
    }

    /// The `k`-th derivative `x⁽ᵏ⁾(0) = k! · c_k`.
    ///
    /// - Precondition: `0 <= k`, and `k <= order` unless the series is a constant.
    @inlinable
    public func derivative(_ k: Int) -> Scalar {
        var result = self[k]
        if k >= 2 {
            for i in 2...k {
                result *= Scalar(i)
            }
        }
        return result
    }

    /// The derivatives `x(0), x'(0), …, x⁽ᴺ⁾(0)`, where `N` is the ``order``.
    @inlinable
    public var derivatives: [Scalar] {
        var result = storage
        var factorial: Scalar = 1
        for k in 1..<result.count {
            factorial *= Scalar(k)
            result[k] *= factorial
        }
        return result
    }

    /// Evaluates the Taylor polynomial `c₀ + c₁ h + … + c_N h^N` at the offset `h`.
    @inlinable
    public func evaluated(at h: Scalar) -> Scalar {
        var result = storage[storage.count - 1]
        for c in storage.dropLast().reversed() {
            result = result * h + c
        }
        return result
    }
}

// MARK: - Internal helpers

extension Taylor {
    /// Whether all coefficients beyond the value are zero.
    ///
    /// Such a series is a constant path. Nonlinear functions map it to a constant path
    /// without running their recurrences, which keeps functions that are singular at
    /// the value (like `sqrt` at zero) from turning `0 · ∞` into NaN.
    @inlinable
    internal var hasZeroHigherCoefficients: Bool {
        storage.dropFirst().allSatisfy { $0 == .zero }
    }

    /// The constant path through `value`, with as many coefficients as `self`.
    @inlinable
    internal func constantPath(_ value: Scalar) -> Taylor {
        var result = [Scalar](repeating: .zero, count: storage.count)
        result[0] = value
        return Taylor(unchecked: result)
    }

    /// The number of coefficients in the result of a binary operation.
    ///
    /// - Precondition: The orders match, or one of the operands is a constant.
    @inlinable
    internal static func commonCount(_ lhs: Taylor, _ rhs: Taylor) -> Int {
        let (m, n) = (lhs.storage.count, rhs.storage.count)
        if m == n || n == 1 {
            return m
        }
        precondition(
            m == 1,
            "Cannot combine Taylor series of different orders (\(m - 1) and \(n - 1))")
        return n
    }

    /// The coefficients as an array of length `count`, padding constants with zeros.
    ///
    /// - Precondition: The series has `count` coefficients, or is a constant.
    @inlinable
    internal func coefficients(count: Int) -> [Scalar] {
        if storage.count == count {
            return storage
        }
        precondition(
            isConstant,
            "Cannot combine Taylor series of different orders (\(order) and \(count - 1))")
        var result = [Scalar](repeating: .zero, count: count)
        result[0] = storage[0]
        return result
    }
}

// MARK: - Equatable

extension Taylor: Equatable {
    /// Whether two series have the same coefficients, where the shorter one is padded
    /// with zeros.
    ///
    /// A constant thus equals a constant path of any order, and more generally a series
    /// equals its extension by zero coefficients. Padding makes the relation transitive.
    @inlinable
    public static func == (lhs: Taylor, rhs: Taylor) -> Bool {
        let (shorter, longer) =
            lhs.storage.count <= rhs.storage.count
            ? (lhs.storage, rhs.storage) : (rhs.storage, lhs.storage)
        return shorter.elementsEqual(longer.prefix(shorter.count))
            && longer.dropFirst(shorter.count).allSatisfy { $0 == .zero }
    }
}

// MARK: - CustomStringConvertible

extension Taylor: CustomStringConvertible {
    /// A textual representation such as `"1.0 + 2.0 t - 0.5 t^2 + O(t^3)"`.
    public var description: String {
        var result = "\(storage[0])"
        for (k, c) in storage.enumerated().dropFirst() {
            result += c.sign == .minus ? " - \(-c)" : " + \(c)"
            result += k == 1 ? " t" : " t^\(k)"
        }
        if !isConstant {
            result += " + O(t^\(storage.count))"
        }
        return result
    }
}
