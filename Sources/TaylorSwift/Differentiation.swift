import RealModule

// MARK: - Functions of one variable

/// The normalized Taylor coefficients `f⁽ᵏ⁾(x) / k!` for `k = 0, …, order`.
///
///     taylorCoefficients(of: { exp($0) }, at: 0.0, order: 3)  // [1, 1, 1/2, 1/6]
///
/// - Precondition: `order >= 0`.
@inlinable
public func taylorCoefficients<Scalar: Real>(
    of f: (Taylor<Scalar>) throws -> Taylor<Scalar>,
    at x: Scalar,
    order: Int
) rethrows -> [Scalar] {
    precondition(order >= 0, "The order of differentiation must not be negative")
    return try f(.variable(x, order: order)).coefficients(count: order + 1)
}

/// The derivatives `f(x), f'(x), …, f⁽ᴺ⁾(x)` up to and including the order `N`.
///
///     derivatives(of: { sin($0) }, at: 0.0, through: 3)  // [0, 1, 0, -1]
///
/// - Precondition: `order >= 0`.
@inlinable
public func derivatives<Scalar: Real>(
    of f: (Taylor<Scalar>) throws -> Taylor<Scalar>,
    at x: Scalar,
    through order: Int
) rethrows -> [Scalar] {
    let coefficients = try taylorCoefficients(of: f, at: x, order: order)
    return Taylor(unchecked: coefficients).derivatives
}

/// The derivative `f⁽ᵏ⁾(x)` of the given order `k`.
///
///     derivative(of: { $0 * $0 * $0 }, at: 2.0, order: 2)  // 12
///
/// - Precondition: `order >= 0`.
@inlinable
public func derivative<Scalar: Real>(
    of f: (Taylor<Scalar>) throws -> Taylor<Scalar>,
    at x: Scalar,
    order: Int = 1
) rethrows -> Scalar {
    let coefficients = try taylorCoefficients(of: f, at: x, order: order)
    return Taylor(unchecked: coefficients).derivative(order)
}

// MARK: - Functions of several variables

/// The normalized Taylor coefficients of `t ↦ f(x + t · direction)` for `k = 0, …, order`.
///
/// The `k`-th coefficient is `1 / k!` times the `k`-th directional derivative of `f`.
///
/// - Precondition: `order >= 0`, and `x` and `direction` have the same number of elements.
@inlinable
public func taylorCoefficients<Scalar: Real>(
    of f: ([Taylor<Scalar>]) throws -> Taylor<Scalar>,
    at x: [Scalar],
    along direction: [Scalar],
    order: Int
) rethrows -> [Scalar] {
    precondition(order >= 0, "The order of differentiation must not be negative")
    precondition(
        x.count == direction.count,
        "The point and the direction must have the same number of elements")
    let path = zip(x, direction).map { Taylor.variable($0, direction: $1, order: order) }
    return try f(path).coefficients(count: order + 1)
}

/// The directional derivative of the given order `k`: the `k`-th derivative of
/// `t ↦ f(x + t · direction)` at `t = 0`.
///
/// For `k = 1` this is the Jacobian-vector product `∇f(x) · v`, for `k = 2` the quadratic
/// form `vᵀ H(x) v` of the Hessian.
///
/// - Precondition: `order >= 0`, and `x` and `direction` have the same number of elements.
@inlinable
public func directionalDerivative<Scalar: Real>(
    of f: ([Taylor<Scalar>]) throws -> Taylor<Scalar>,
    at x: [Scalar],
    along direction: [Scalar],
    order: Int = 1
) rethrows -> Scalar {
    let coefficients = try taylorCoefficients(of: f, at: x, along: direction, order: order)
    return Taylor(unchecked: coefficients).derivative(order)
}

/// The gradient `∇f(x)`, from one first-order pass per variable.
@inlinable
public func gradient<Scalar: Real>(
    of f: ([Taylor<Scalar>]) throws -> Taylor<Scalar>,
    at x: [Scalar]
) rethrows -> [Scalar] {
    var direction = [Scalar](repeating: .zero, count: x.count)
    var result = [Scalar](repeating: .zero, count: x.count)
    for i in x.indices {
        direction[i] = 1
        result[i] = try taylorCoefficients(of: f, at: x, along: direction, order: 1)[1]
        direction[i] = .zero
    }
    return result
}

/// The Hessian matrix of second partial derivatives, as an array of rows.
///
/// Taylor mode computes second derivatives along one direction at a time. The diagonal
/// follows from the coordinate directions `eᵢ`, and the mixed derivatives follow from the
/// directions `eᵢ + eⱼ` through the polarization identity
///
///     2 Hᵢⱼ = (eᵢ + eⱼ)ᵀ H (eᵢ + eⱼ) - eᵢᵀ H eᵢ - eⱼᵀ H eⱼ.
///
/// This takes `n (n + 1) / 2` second-order passes for `n` variables.
@inlinable
public func hessian<Scalar: Real>(
    of f: ([Taylor<Scalar>]) throws -> Taylor<Scalar>,
    at x: [Scalar]
) rethrows -> [[Scalar]] {
    let n = x.count
    var direction = [Scalar](repeating: .zero, count: n)

    // The second-order coefficients `eᵢᵀ H eᵢ / 2` along the coordinate directions.
    var pure = [Scalar](repeating: .zero, count: n)
    for i in 0..<n {
        direction[i] = 1
        pure[i] = try taylorCoefficients(of: f, at: x, along: direction, order: 2)[2]
        direction[i] = .zero
    }

    var result = [[Scalar]](repeating: [Scalar](repeating: .zero, count: n), count: n)
    for i in 0..<n {
        result[i][i] = 2 * pure[i]
        for j in (i + 1)..<n {
            direction[i] = 1
            direction[j] = 1
            let mixed = try taylorCoefficients(of: f, at: x, along: direction, order: 2)[2]
            direction[i] = .zero
            direction[j] = .zero
            result[i][j] = mixed - pure[i] - pure[j]
            result[j][i] = result[i][j]
        }
    }
    return result
}

/// The Laplacian `Σᵢ ∂²f/∂xᵢ²`, from one second-order pass per variable.
@inlinable
public func laplacian<Scalar: Real>(
    of f: ([Taylor<Scalar>]) throws -> Taylor<Scalar>,
    at x: [Scalar]
) rethrows -> Scalar {
    var direction = [Scalar](repeating: .zero, count: x.count)
    var result = Scalar.zero
    for i in x.indices {
        direction[i] = 1
        result += 2 * (try taylorCoefficients(of: f, at: x, along: direction, order: 2)[2])
        direction[i] = .zero
    }
    return result
}

/// A mixed partial derivative of arbitrary order.
///
/// `variables` lists the index of one variable per differentiation, with repetition:
///
///     // ∂³f / ∂x₀² ∂x₁
///     mixedPartial(of: f, at: x, withRespectTo: [0, 0, 1])
///
/// Taylor mode computes derivatives along one direction at a time. A mixed partial of
/// order `d` is recovered exactly from `d`-th order directional derivatives along signed
/// sums of coordinate directions, through the polarization identity
///
///     ∂^d f / ∂x_{i₁} ⋯ ∂x_{i_d} = 1 / (2^d d!) Σ_ε ε₁ ⋯ ε_d D^d f [Σ_k ε_k e_{i_k}],
///
/// where the sum runs over all signs `ε ∈ {±1}^d` and `D^d f [v]` is the `d`-th
/// directional derivative along `v`. This takes at most `2^(d-1)` passes of order `d`.
///
/// - Precondition: Every element of `variables` is a valid index of `x`.
@inlinable
public func mixedPartial<Scalar: Real>(
    of f: ([Taylor<Scalar>]) throws -> Taylor<Scalar>,
    at x: [Scalar],
    withRespectTo variables: [Int]
) rethrows -> Scalar {
    precondition(
        variables.allSatisfy { x.indices.contains($0) },
        "Cannot differentiate with respect to a variable that is out of bounds")
    let order = variables.count
    guard order > 0 else {
        return try f(x.map { Taylor($0) }).value
    }
    precondition(order < Int.bitWidth - 1, "The order of differentiation is too high")

    // The terms for `ε` and `-ε` are equal, so that fixing `ε₁ = +1` halves the work.
    // Sign patterns that lead to the same direction are merged into a single weight.
    var weights: [[Int]: Int] = [:]
    for signs in 0..<(1 << (order - 1)) {
        var direction = [Int](repeating: 0, count: x.count)
        var weight = 1
        for (k, variable) in variables.enumerated() {
            if k > 0 && (signs >> (k - 1)) & 1 == 1 {
                direction[variable] -= 1
                weight = -weight
            } else {
                direction[variable] += 1
            }
        }
        weights[direction, default: 0] += weight
    }

    // Sum in a fixed order to keep the result reproducible.
    let terms = weights.sorted { $0.key.lexicographicallyPrecedes($1.key) }
    var sum = Scalar.zero
    for (direction, weight) in terms {
        // The directional derivative along the zero vector vanishes.
        if weight == 0 || direction.allSatisfy({ $0 == 0 }) {
            continue
        }
        let coefficients = try taylorCoefficients(
            of: f, at: x, along: direction.map { Scalar($0) }, order: order)
        sum += Scalar(weight) * coefficients[order]
    }
    // `D^d f [v]` is `d!` times the `d`-th Taylor coefficient, which cancels the factorial.
    return sum / Scalar(1 << (order - 1))
}
