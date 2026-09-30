import RealModule
import Testing
import TaylorSwift

/// Tests against closed-form Taylor expansions.
@Suite struct ClosedFormTests {
    /// The coefficients `1 / k!`.
    func inverseFactorials(through order: Int) -> [Double] {
        var result = [1.0]
        for k in 0..<order {
            result.append(result[k] / Double(k + 1))
        }
        return result
    }

    /// The coefficients `binom(r, k) x^(r-k)` of `(x + t)^r`.
    func binomialSeries(_ x: Double, _ r: Double, through order: Int) -> [Double] {
        var result = [Double.pow(x, r)]
        for k in 0..<order {
            result.append(result[k] * (r - Double(k)) / (Double(k + 1) * x))
        }
        return result
    }

    @Test func exponential() {
        let x = Taylor.variable(0.0, order: 25)
        expectClose(exp(x).coefficients, inverseFactorials(through: 25), tolerance: 1e-15)

        // Every derivative of exp is exp.
        let y = Taylor.variable(1.5, order: 6)
        expectClose(exp(y).derivatives, [Double](repeating: Double.exp(1.5), count: 7))
    }

    @Test func exponentialMinusOne() {
        let x = Taylor.variable(1e-10, order: 3)
        let y = expm1(x)
        #expect(y.value == Double.expMinusOne(1e-10))
        expectClose(Array(y.derivatives.dropFirst()), [Double](repeating: Double.exp(1e-10), count: 3))
    }

    @Test func logarithm() {
        // log(x + t) has the coefficients (-1)^(k-1) / (k x^k).
        let x0 = 1.5
        var expected = [Double.log(x0)]
        for k in 1...8 {
            expected.append((k % 2 == 1 ? 1 : -1) / (Double(k) * Double.pow(x0, Double(k))))
        }
        expectClose(log(Taylor.variable(x0, order: 8)).coefficients, expected)
    }

    @Test func logarithmOfOnePlus() {
        // log(1 + t) = t - t²/2 + t³/3 - …
        let x = Taylor.variable(0.0, order: 5)
        expectClose(log1p(x).coefficients, [0, 1, -1.0 / 2, 1.0 / 3, -1.0 / 4, 1.0 / 5])

        let y = log1p(Taylor.variable(1e-10, order: 1))
        #expect(y.value == Double.log(onePlus: 1e-10))
    }

    @Test func sineAndCosine() {
        let x0 = 0.7
        let (s, c) = (Double.sin(x0), Double.cos(x0))
        let x = Taylor.variable(x0, order: 8)
        expectClose(sin(x).derivatives, [s, c, -s, -c, s, c, -s, -c, s])
        expectClose(cos(x).derivatives, [c, -s, -c, s, c, -s, -c, s, c])
    }

    @Test func tangent() {
        let x = Taylor.variable(0.0, order: 7)
        expectClose(tan(x).coefficients, [0, 1, 0, 1.0 / 3, 0, 2.0 / 15, 0, 17.0 / 315])

        // tan' = 1 + tan², tan'' = 2 tan (1 + tan²)
        let t = Double.tan(0.4)
        let y = tan(Taylor.variable(0.4, order: 2))
        expectClose(y.derivatives, [t, 1 + t * t, 2 * t * (1 + t * t)])
    }

    @Test func hyperbolicSineAndCosine() {
        let x0 = 0.7
        let (s, c) = (Double.sinh(x0), Double.cosh(x0))
        let x = Taylor.variable(x0, order: 5)
        expectClose(sinh(x).derivatives, [s, c, s, c, s, c])
        expectClose(cosh(x).derivatives, [c, s, c, s, c, s])
    }

    @Test func hyperbolicTangent() {
        let x = Taylor.variable(0.0, order: 7)
        expectClose(tanh(x).coefficients, [0, 1, 0, -1.0 / 3, 0, 2.0 / 15, 0, -17.0 / 315])

        // tanh' = 1 - tanh², tanh'' = -2 tanh (1 - tanh²)
        let t = Double.tanh(0.4)
        let y = tanh(Taylor.variable(0.4, order: 2))
        expectClose(y.derivatives, [t, 1 - t * t, -2 * t * (1 - t * t)])
    }

    @Test func inverseSineAndCosine() {
        let x = Taylor.variable(0.0, order: 7)
        let expected = [0, 1, 0, 1.0 / 6, 0, 3.0 / 40, 0, 5.0 / 112]
        expectClose(asin(x).coefficients, expected)
        expectClose(acos(x).coefficients, [Double.pi / 2] + expected.dropFirst().map { -$0 })
    }

    @Test func inverseTangent() {
        let x = Taylor.variable(0.0, order: 7)
        expectClose(atan(x).coefficients, [0, 1, 0, -1.0 / 3, 0, 1.0 / 5, 0, -1.0 / 7])
    }

    @Test func inverseHyperbolicFunctions() {
        let x = Taylor.variable(0.0, order: 7)
        expectClose(asinh(x).coefficients, [0, 1, 0, -1.0 / 6, 0, 3.0 / 40, 0, -5.0 / 112])
        expectClose(atanh(x).coefficients, [0, 1, 0, 1.0 / 3, 0, 1.0 / 5, 0, 1.0 / 7])

        // acosh' = (x² - 1)^(-1/2), acosh'' = -x (x² - 1)^(-3/2),
        // acosh''' = (2x² + 1) (x² - 1)^(-5/2)
        let x0 = 2.0
        let d = x0 * x0 - 1
        expectClose(
            acosh(Taylor.variable(x0, order: 3)).derivatives,
            [
                Double.acosh(x0),
                Double.pow(d, -0.5),
                -x0 * Double.pow(d, -1.5),
                (2 * x0 * x0 + 1) * Double.pow(d, -2.5),
            ])
    }

    @Test func squareRoot() {
        let x = Taylor.variable(2.0, order: 8)
        expectClose(sqrt(x).coefficients, binomialSeries(2, 0.5, through: 8))
    }

    @Test func roots() {
        let x = Taylor.variable(2.0, order: 8)
        expectClose(cbrt(x).coefficients, binomialSeries(2, 1.0 / 3, through: 8))
        expectClose(Taylor.root(x, 5).coefficients, binomialSeries(2, 0.2, through: 8))
        #expect(Taylor.root(x, 1) == x)

        // Odd roots of negative values are real: ∛(-2 + t) = -∛(2 - t)
        let y = Taylor.variable(-2.0, order: 8)
        let mirrored = cbrt(Taylor.variable(2.0, direction: -1, order: 8))
        expectClose(cbrt(y).coefficients, (-mirrored).coefficients)
        #expect(Taylor.root(y, 2).value.isNaN)
    }

    @Test func powers() {
        let x = Taylor.variable(2.0, order: 8)
        expectClose(pow(x, 2.5).coefficients, binomialSeries(2, 2.5, through: 8))
        expectClose(pow(x, -1.5).coefficients, binomialSeries(2, -1.5, through: 8))
        expectClose(pow(x, 3).coefficients, [8, 12, 6, 1, 0, 0, 0, 0, 0])
        expectClose(pow(x, -2).coefficients, binomialSeries(2, -2, through: 8))
        expectClose(pow(x, Taylor<Double>(2.5)).coefficients, binomialSeries(2, 2.5, through: 8))

        let exponent = 2.5
        expectClose(pow(x, exponent).coefficients, binomialSeries(2, 2.5, through: 8))
        expectClose(Taylor.pow(x, exponent).coefficients, binomialSeries(2, 2.5, through: 8))
    }

    @Test func exponentialWithOtherBases() {
        // d^k/dx^k b^x = log(b)^k b^x
        let x = Taylor.variable(1.5, order: 4)
        let ln2 = Double.log(2.0)
        let ln10 = Double.log(10.0)
        expectClose(exp2(x).derivatives, (0...4).map { Double.pow(ln2, Double($0)) * Double.exp2(1.5) })
        expectClose(
            Taylor.exp10(x).derivatives,
            (0...4).map { Double.pow(ln10, Double($0)) * Double.pow(10, 1.5) })
        expectClose(pow(Taylor<Double>(2.0), x).derivatives, exp2(x).derivatives)
    }

    @Test func logarithmWithOtherBases() {
        let x = Taylor.variable(1.5, order: 4)
        expectClose(log2(x).coefficients, (log(x) / Double.log(2.0)).coefficients)
        expectClose(log10(x).coefficients, (log(x) / Double.log(10.0)).coefficients)
        #expect(log2(Taylor.variable(8.0, order: 1)).value == 3)
        #expect(log10(Taylor.variable(1000.0, order: 1)).value == 3)
    }

    @Test func errorFunction() {
        // erf(t) = 2/√π (t - t³/3 + t⁵/10 - t⁷/42 + …)
        let scale = 2 / Double.pi.squareRoot()
        let series = [0, 1, 0, -1.0 / 3, 0, 1.0 / 10, 0, -1.0 / 42].map { scale * $0 }
        let x = Taylor.variable(0.0, order: 7)
        expectClose(erf(x).coefficients, series)
        expectClose(erfc(x).coefficients, [1] + series.dropFirst().map { -$0 })

        // erf' = 2/√π exp(-x²), erf'' = -2x erf'
        let x0 = 0.8
        let d = scale * Double.exp(-x0 * x0)
        expectClose(erf(Taylor.variable(x0, order: 2)).derivatives, [Double.erf(x0), d, -2 * x0 * d])
    }

    @Test func absoluteValue() {
        #expect(abs(Taylor.variable(2.0, order: 2)).coefficients == [2, 1, 0])
        #expect(abs(Taylor.variable(-2.0, order: 2)).coefficients == [2, -1, 0])
        #expect(abs(Taylor.variable(0.0, order: 2)).coefficients == [0, 1, 0])
    }

    @Test func composition() {
        // f = exp(sin(x)): f' = c f, f'' = (c² - s) f, f''' = (c³ - 3cs - c) f
        let x0 = 0.9
        let (s, c) = (Double.sin(x0), Double.cos(x0))
        let f = Double.exp(s)
        let y = exp(sin(Taylor.variable(x0, order: 3)))
        expectClose(y.derivatives, [f, c * f, (c * c - s) * f, (c * c * c - 3 * c * s - c) * f])
    }

    @Test func chainRuleAlongACurvedPath() {
        // Faà di Bruno: the coefficients of f(u(t)) are
        //   f' u₁,  f' u₂ + f'' u₁² / 2,  f' u₃ + f'' u₁ u₂ + f''' u₁³ / 6.
        let u = Taylor(coefficients: [0.7, 0.3, -0.2, 0.5])
        let (u0, u1, u2, u3) = (u[0], u[1], u[2], u[3])
        func expected(_ f: Double, _ f1: Double, _ f2: Double, _ f3: Double) -> [Double] {
            [f, f1 * u1, f1 * u2 + f2 * u1 * u1 / 2, f1 * u3 + f2 * u1 * u2 + f3 * u1 * u1 * u1 / 6]
        }

        let e = Double.exp(u0)
        expectClose(exp(u).coefficients, expected(e, e, e, e))

        let (s, c) = (Double.sin(u0), Double.cos(u0))
        expectClose(sin(u).coefficients, expected(s, c, -s, -c))
        expectClose(cos(u).coefficients, expected(c, -s, -c, s))

        expectClose(
            log(u).coefficients,
            expected(Double.log(u0), 1 / u0, -1 / (u0 * u0), 2 / (u0 * u0 * u0)))

        let r = u0.squareRoot()
        expectClose(
            sqrt(u).coefficients,
            expected(r, 0.5 / r, -0.25 / (r * u0), 0.375 / (r * u0 * u0)))

        let t = Double.tan(u0)
        let sec2 = 1 + t * t
        expectClose(
            tan(u).coefficients,
            expected(t, sec2, 2 * t * sec2, 2 * sec2 * sec2 + 4 * t * t * sec2))

        let a = 1 + u0 * u0
        expectClose(
            atan(u).coefficients,
            expected(Double.atan(u0), 1 / a, -2 * u0 / (a * a), (6 * u0 * u0 - 2) / (a * a * a)))
    }
}

/// Tests of functional identities along curved paths.
@Suite struct IdentityTests {
    let u = Taylor(coefficients: [0.7, 0.3, -0.2, 0.5, 0.1, -0.4, 0.25])
    let w = Taylor(coefficients: [1.3, -0.6, 0.4, 0.2, -0.3, 0.15, 0.05])
    let one = [1.0, 0, 0, 0, 0, 0, 0]
    let tolerance = 1e-10

    @Test func exponentialAndLogarithm() {
        expectClose(exp(log(u)).coefficients, u.coefficients, tolerance: tolerance)
        expectClose(log(exp(u)).coefficients, u.coefficients, tolerance: tolerance)
        expectClose(log(u * w).coefficients, (log(u) + log(w)).coefficients, tolerance: tolerance)
        expectClose(exp(u + w).coefficients, (exp(u) * exp(w)).coefficients, tolerance: tolerance)
        expectClose(expm1(u).coefficients, (exp(u) - 1).coefficients, tolerance: tolerance)
        expectClose(log1p(u).coefficients, log(1 + u).coefficients, tolerance: tolerance)
        expectClose(log2(exp2(u)).coefficients, u.coefficients, tolerance: tolerance)
        expectClose(
            Taylor.log10(Taylor.exp10(u)).coefficients, u.coefficients, tolerance: tolerance)
    }

    @Test func trigonometricFunctions() {
        expectClose((sin(u) * sin(u) + cos(u) * cos(u)).coefficients, one, tolerance: tolerance)
        expectClose(tan(u).coefficients, (sin(u) / cos(u)).coefficients, tolerance: tolerance)
        expectClose(sin(u + w).coefficients, (sin(u) * cos(w) + cos(u) * sin(w)).coefficients, tolerance: tolerance)
    }

    @Test func hyperbolicFunctions() {
        expectClose((cosh(u) * cosh(u) - sinh(u) * sinh(u)).coefficients, one, tolerance: tolerance)
        expectClose(tanh(u).coefficients, (sinh(u) / cosh(u)).coefficients, tolerance: tolerance)
        expectClose(sinh(u).coefficients, ((exp(u) - exp(-u)) / 2).coefficients, tolerance: tolerance)
        expectClose(cosh(u).coefficients, ((exp(u) + exp(-u)) / 2).coefficients, tolerance: tolerance)
    }

    @Test func inverseFunctions() {
        expectClose(sin(asin(u)).coefficients, u.coefficients, tolerance: tolerance)
        expectClose(cos(acos(u)).coefficients, u.coefficients, tolerance: tolerance)
        expectClose(tan(atan(u)).coefficients, u.coefficients, tolerance: tolerance)
        expectClose(sinh(asinh(u)).coefficients, u.coefficients, tolerance: tolerance)
        expectClose(cosh(acosh(w)).coefficients, w.coefficients, tolerance: tolerance)
        expectClose(tanh(atanh(u)).coefficients, u.coefficients, tolerance: tolerance)
        expectClose(asin(sin(u)).coefficients, u.coefficients, tolerance: tolerance)
        expectClose(acos(cos(u)).coefficients, u.coefficients, tolerance: tolerance)
        expectClose(atan(tan(u)).coefficients, u.coefficients, tolerance: tolerance)
        expectClose(asinh(sinh(u)).coefficients, u.coefficients, tolerance: tolerance)
        expectClose(acosh(cosh(u)).coefficients, u.coefficients, tolerance: tolerance)
        expectClose(atanh(tanh(u)).coefficients, u.coefficients, tolerance: tolerance)
    }

    @Test func rootsAndPowers() {
        expectClose((sqrt(u) * sqrt(u)).coefficients, u.coefficients, tolerance: tolerance)
        expectClose(sqrt(u * u).coefficients, u.coefficients, tolerance: tolerance)
        let root = cbrt(u)
        expectClose((root * root * root).coefficients, u.coefficients, tolerance: tolerance)
        expectClose(cbrt(-u).coefficients, (-root).coefficients, tolerance: tolerance)
        expectClose(Taylor.root(u, 2).coefficients, sqrt(u).coefficients, tolerance: tolerance)
        expectClose(Taylor.root(u, -2).coefficients, (1 / sqrt(u)).coefficients, tolerance: tolerance)

        expectClose(pow(u, 3).coefficients, (u * u * u).coefficients, tolerance: tolerance)
        expectClose(pow(u, -2).coefficients, (1 / (u * u)).coefficients, tolerance: tolerance)
        expectClose(pow(u, 0).coefficients, one, tolerance: tolerance)
        expectClose(pow(u, 1).coefficients, u.coefficients, tolerance: tolerance)
        expectClose(pow(u, 0.5).coefficients, sqrt(u).coefficients, tolerance: tolerance)
        expectClose(pow(u, 3.0).coefficients, (u * u * u).coefficients, tolerance: tolerance)
        expectClose(pow(u, -1.0).coefficients, (1 / u).coefficients, tolerance: tolerance)
        expectClose(pow(u, w).coefficients, exp(w * log(u)).coefficients, tolerance: tolerance)
        expectClose(pow(u, w + 1).coefficients, (pow(u, w) * u).coefficients, tolerance: tolerance)

        // Integer powers of negative values.
        expectClose(pow(-u, 3).coefficients, (-(u * u * u)).coefficients, tolerance: tolerance)
        expectClose(pow(-u, -2).coefficients, (1 / (u * u)).coefficients, tolerance: tolerance)
    }

    @Test func euclideanNorm() {
        expectClose(hypot(u, w).coefficients, sqrt(u * u + w * w).coefficients, tolerance: tolerance)
        expectClose(hypot(u, Taylor<Double>(0.0)).coefficients, u.coefficients, tolerance: tolerance)
    }

    @Test func errorFunctions() {
        expectClose((erf(u) + erfc(u)).coefficients, one, tolerance: tolerance)
        expectClose(erf(-u).coefficients, (-erf(u)).coefficients, tolerance: tolerance)
    }

    @Test func twoArgumentInverseTangent() {
        // In the right half-plane, atan2 is atan(y / x).
        expectClose(atan2(u, w).coefficients, atan(u / w).coefficients, tolerance: tolerance)

        // The angle of the point r (cos φ, sin φ) is φ, in all four quadrants.
        for angle in [0.4, 2.5, -2.5, -0.4, Double.pi / 2, -Double.pi / 2] {
            let phi = u - u.value + angle
            let actual = atan2(w * sin(phi), w * cos(phi))
            expectClose(actual.coefficients, phi.coefficients, tolerance: tolerance)
        }

        // Constants are lifted to the order of the other argument.
        expectClose(atan2(u, Taylor<Double>(1.0)).coefficients, atan(u).coefficients, tolerance: tolerance)
        expectClose(
            atan2(Taylor<Double>(1.0), u).coefficients,
            (Double.pi / 2 - atan(u)).coefficients,
            tolerance: tolerance)
    }
}

/// Tests at singular points and with constants.
@Suite struct EdgeCaseTests {
    @Test func constantsStayConstants() {
        let c = Taylor<Double>(0.5)
        #expect(exp(c).coefficients == [Double.exp(0.5)])
        #expect(sin(c).coefficients == [Double.sin(0.5)])
        #expect(sqrt(c).coefficients == [(0.5).squareRoot()])
        #expect(pow(c, 2).coefficients == [0.25])
        #expect(pow(c, c).coefficients == [(0.5).squareRoot()])
        #expect(atan2(c, c).coefficients == [Double.pi / 4])
    }

    @Test func constantPathsAvoidSingularities() {
        // The derivative of √ is infinite at zero, but a constant path never multiplies by it.
        let zero = Taylor.variable(0.0, direction: 0, order: 3)
        #expect(sqrt(zero).coefficients == [0, 0, 0, 0])
        #expect(cbrt(zero).coefficients == [0, 0, 0, 0])
        #expect(pow(zero, 0.5).coefficients == [0, 0, 0, 0])
        #expect(asin(zero + 1).coefficients == [Double.pi / 2, 0, 0, 0])
        #expect(acos(zero + 1).coefficients == [0, 0, 0, 0])
        #expect(acosh(zero + 1).coefficients == [0, 0, 0, 0])
        #expect(hypot(zero, zero).coefficients == [0, 0, 0, 0])
    }

    @Test func singularitiesAreNotFinite() {
        let x = Taylor.variable(0.0, order: 2)
        #expect(sqrt(x).value == 0)
        #expect(sqrt(x)[1] == .infinity)
        #expect(!sqrt(x)[2].isFinite)
        #expect(log(x).value == -.infinity)
        #expect(!log(x)[1].isFinite)
        #expect(!pow(x, 0.5)[1].isFinite)
    }

    @Test func integerPowersAtZero() {
        let x = Taylor.variable(0.0, order: 4)
        #expect(pow(x, 3).coefficients == [0, 0, 0, 1, 0])
        #expect(pow(x, 3.0).coefficients == [0, 0, 0, 1, 0])
        #expect(pow(x, Taylor<Double>(3.0)).coefficients == [0, 0, 0, 1, 0])
        #expect(pow(x, 1).coefficients == [0, 1, 0, 0, 0])
        #expect(pow(x, 0).coefficients == [1, 0, 0, 0, 0])
        #expect(pow(x, 7).coefficients == [0, 0, 0, 0, 0])

        let exponent = 3.0
        #expect(pow(x, exponent).coefficients == [0, 0, 0, 1, 0])
    }

    @Test func integerPowersOfNegativeValues() {
        // (-2 + t)³ = -8 + 12t - 6t² + t³
        let x = Taylor.variable(-2.0, order: 3)
        #expect(pow(x, 3).coefficients == [-8, 12, -6, 1])
        // Real exponents follow the scalar `pow`, which is NaN for negative bases.
        let exponent = 3.0
        #expect(pow(x, exponent).value.isNaN)
    }

    @Test func zeroToAPositivePower() {
        let zero = Taylor.variable(0.0, direction: 0, order: 2)
        let y = Taylor.variable(2.0, order: 2)
        #expect(pow(zero, y).coefficients == [0, 0, 0])
    }

    @Test func orderZero() {
        let x = Taylor.variable(0.5, order: 0)
        #expect(exp(x).coefficients == [Double.exp(0.5)])
        #expect(tan(x).coefficients == [Double.tan(0.5)])
        #expect(atan2(x, x).coefficients == [Double.pi / 4])
        #expect(erf(x).coefficients == [Double.erf(0.5)])
    }

    @Test func nanPropagates() {
        let x = Taylor(coefficients: [1.0, .nan, 0])
        #expect(exp(x)[1].isNaN)
        #expect(sqrt(x)[1].isNaN)
    }

    @Test func floatScalars() {
        let x = Taylor<Float>.variable(0, order: 3)
        let sixth: Float = 1 / 6
        #expect(exp(x).coefficients == [1, 1, 0.5, sixth])
        #expect(sin(x).coefficients == [0, 1, 0, -sixth])
        #expect(pow(x + 1, 2.0).coefficients == [1, 2, 1, 0])
    }
}
