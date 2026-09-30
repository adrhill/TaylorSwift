import Testing
import TaylorSwift
import RealModule

// Functions that are not smooth at zero pick the expansion for `t → 0⁺` there. It is exact
// where the composite path is smooth, and one-sided otherwise.

@Suite struct AbsoluteValueAtZeroTests {
    let t = Taylor.variable(0.0, order: 3)

    @Test func smoothCasesAreExact() {
        // |−t²| = t², however the zero value comes about.
        #expect(abs(-(t * t)).coefficients == [0, 0, 1, 0])
        #expect(abs(0.0 - t * t).coefficients == [0, 0, 1, 0])
        #expect(abs(t * t - 2 * (t * t)).coefficients == [0, 0, 1, 0])
        #expect(abs(t * t).coefficients == [0, 0, 1, 0])
        // |t³ − t²| = t² − t³ near zero.
        #expect(abs(t * t * t - t * t).coefficients == [0, 0, 1, -1])
    }

    @Test func kinksGiveTheRightSidedExpansion() {
        #expect(abs(t).coefficients == [0, 1, 0, 0])
        #expect(abs(-t).coefficients == [0, 1, 0, 0])
        // |t³ − t| = t − t³ for small t > 0.
        #expect(abs(t * t * t - t).coefficients == [0, 1, 0, -1])
    }

    @Test func signOfZeroDoesNotMatter() {
        for direction in [1.0, -1.0] {
            let positive = abs(Taylor.variable(0.0, direction: direction, order: 2))
            let negative = abs(Taylor.variable(-0.0, direction: direction, order: 2))
            #expect(positive.coefficients == [0, 1, 0])
            #expect(negative.coefficients == [0, 1, 0])
        }
        #expect(abs(Taylor<Double>(-0.0)).value.sign == .plus)
        let zeroPath = abs(Taylor(coefficients: [-0.0, 0, 0]))
        #expect(zeroPath.value.sign == .plus && zeroPath.coefficients == [0, 0, 0])
    }

    @Test func rightDerivativesAtAKink() {
        // |x² − 1| is 1 − x² inside and x² − 1 outside of [−1, 1].
        #expect(derivatives(of: { abs($0 * $0 - 1) }, at: 1.0, through: 3) == [0, 2, 2, 0])
        #expect(derivatives(of: { abs($0 * $0 - 1) }, at: -1.0, through: 3) == [0, 2, -2, 0])
    }

    @Test func awayFromZero() {
        let x = Taylor(coefficients: [-0.5, 0, 2])
        #expect(abs(x).coefficients == [0.5, 0, -2])
        #expect(abs(-x).coefficients == [0.5, 0, -2])
    }
}

@Suite struct PowersAtZeroTests {
    let t = Taylor.variable(0.0, order: 4)
    let inf = Double.infinity
    let nan = Double.nan

    @Test func squareRootOfALinearPath() {
        // (√t)⁽ᵏ⁾ = (1/2)(−1/2)⋯(3/2 − k) t^(1/2 − k) diverges with alternating signs.
        let expected = [0, inf, -inf, inf, -inf]
        expectClose(sqrt(t).coefficients, expected)
        expectClose(sqrt(2 * t).coefficients, expected)
        expectClose(pow(t, 0.5).coefficients, expected)
        expectClose(Taylor.root(t, 2).coefficients, expected)
        // √(−t) is not real for t > 0.
        expectClose(sqrt(-t).coefficients, [0, nan, nan, nan, nan])
    }

    @Test func squareRootOfSquares() {
        // √(t²) = t for t > 0. The coefficient of t⁴ would need that of t⁵ in t².
        expectClose(sqrt(t * t).coefficients, [0, 1, 0, 0, nan])
        // √((t + t²)²) = t + t²
        let x = t + t * t
        expectClose(sqrt(x * x).coefficients, [0, 1, 1, 0, nan])
        // √(t² + t³) = t √(1 + t) = t + t²/2 − t³/8 + …
        expectClose(sqrt(t * t + t * t * t).coefficients, [0, 1, 0.5, -0.125, nan])
        // √(2t²) = √2 t
        expectClose(sqrt(2 * t * t).coefficients, [0, 2.0.squareRoot(), 0, 0, nan])
    }

    @Test func squareRootOfAQuarticIsSmooth() {
        // √(4t⁴ + O(t⁶)) = 2t² + O(t⁴): the coefficient of t⁴ depends on that of t⁶.
        let x = Taylor.variable(0.0, order: 5)
        let quartic = 4 * pow(x, 4)
        expectClose(sqrt(quartic).coefficients, [0, 0, 2, 0, nan, nan])
    }

    @Test func fractionalPowers() {
        // t^2.5 has two vanishing derivatives, and diverging ones from the third on.
        expectClose(pow(t, 2.5).coefficients, [0, 0, 0, inf, -inf])
        expectClose(pow(3 * t, 2.5).coefficients, [0, 0, 0, inf, -inf])
        expectClose(pow(t, 1.5).coefficients, [0, 0, inf, -inf, inf])
        expectClose(pow(t, 0.25).coefficients, [0, inf, -inf, inf, -inf])
        // Fractional powers of paths that are negative for t > 0 are not real.
        expectClose(pow(-t, 1.5).coefficients, [0, nan, nan, nan, nan])
        // All derivatives within the truncation order vanish.
        expectClose(pow(t, 10.5).coefficients, [0, 0, 0, 0, 0])
    }

    @Test func fractionalPowersWithAWholeLeadingOrder() {
        // (t²)^1.5 = t³ for t > 0.
        expectClose(pow(t * t, 1.5).coefficients, [0, 0, 0, 1, 0])
        // (t² + t³)^1.5 = t³ (1 + t)^1.5 = t³ + 1.5 t⁴ + …
        expectClose(pow(t * t + t * t * t, 1.5).coefficients, [0, 0, 0, 1, 1.5])
        // (2t²)^0.5 = √2 t, and the coefficient of t⁴ is not determined.
        expectClose(pow(2 * t * t, 0.5).coefficients, [0, 2.0.squareRoot(), 0, 0, nan])
    }

    @Test func seriesExponentsMatchScalarExponents() {
        for y in [0.5, 1.5, 2.5, 10.5] {
            for x in [t, t * t, t * t + t * t * t, -t] {
                expectClose(pow(x, Taylor<Double>(y)).coefficients, pow(x, y).coefficients)
            }
        }
    }

    @Test func oddRootsOfPathsThroughZero() {
        let expected = [0, inf, -inf, inf, -inf]
        expectClose(cbrt(t).coefficients, expected)
        expectClose(cbrt(-t).coefficients, expected.map { -$0 })

        // Odd roots are odd functions, so ∛(±t³) = ±t on both sides of zero. The coefficient
        // of t³ would need that of t⁵ in t³.
        let cube = t * t * t
        expectClose(cbrt(cube).coefficients, [0, 1, 0, nan, nan])
        expectClose(cbrt(-cube).coefficients, [0, -1, 0, nan, nan])
        expectClose(Taylor.root(-pow(t, 4), 4).coefficients, [0, nan, nan, nan, nan])
    }

    @Test func rootsWithWholeLeadingOrder() {
        // ∛(t⁶ + t⁷) = t² ∛(1 + t) = t² + t³/3 − t⁴/9 + …
        let x = Taylor.variable(0.0, order: 8)
        let y = pow(x, 6) + pow(x, 7)
        expectClose(cbrt(y).coefficients, [0, 0, 1, 1.0 / 3, -1.0 / 9, nan, nan, nan, nan])
        // ⁴√(t⁴) = t for t > 0, and √(t⁴) = t².
        expectClose(Taylor.root(pow(x, 4), 4).coefficients, [0, 1, 0, 0, 0, 0, nan, nan, nan])
        expectClose(sqrt(pow(x, 4)).coefficients, [0, 0, 1, 0, 0, 0, 0, nan, nan])
    }

    @Test func negativeExponentsStaySingular() {
        #expect(pow(t, -0.5).value == inf)
        #expect(!pow(t, -0.5)[1].isFinite)
        #expect(Taylor.root(t, -2).value == inf)
        #expect(!Taylor.root(t, -2)[1].isFinite)
    }

    @Test func throughTheDerivativeAPI() {
        // √(x² + x³) = x √(1 + x) for x > 0.
        let result = derivatives(of: { sqrt($0 * $0 + $0 * $0 * $0) }, at: 0.0, through: 3)
        expectClose(result, [0, 1, 1, nan])
        // x^1.5 at zero: f' = 0, f'' diverges.
        expectClose(derivatives(of: { pow($0, 1.5) }, at: 0.0, through: 2), [0, 0, inf])
    }

    @Test func floatScalars() {
        let x = Taylor<Float>.variable(0, order: 3)
        let result = sqrt(x * x + x * x * x).coefficients
        #expect(result[0] == 0 && result[1] == 1 && result[2] == 0.5 && result[3].isNaN)
        #expect(pow(x, 1.5 as Float).coefficients == [0, 0, .infinity, -.infinity])
    }
}

@Suite struct EuclideanNormTests {
    /// The coefficients of `hypot(a + t, b)`, from `f' = (a + t) / f`, `f'' = b² / f³` and
    /// `f''' = −3 b² (a + t) / f⁵`.
    func expected(_ a: Double, _ b: Double) -> [Double] {
        let h = Double.hypot(a, b)
        return [h, a / h, b * b / (2 * h * h * h), -a * b * b / (2 * h * h * h * h * h)]
    }

    let points: [(Double, Double)] = [(3, 4), (-3, 4), (0.5, -2), (0, 1), (1, 0), (-1, 0), (1e-3, 1)]

    @Test func closedForm() {
        for (a, b) in points {
            let result = hypot(Taylor.variable(a, order: 3), Taylor<Double>(b))
            expectClose(result.coefficients, expected(a, b))
            // The arguments are interchangeable.
            let swapped = hypot(Taylor<Double>(b), Taylor.variable(a, order: 3))
            expectClose(swapped.coefficients, expected(a, b))
        }
    }

    @Test(arguments: [-1000, -600, -300, -200, 200, 300, 600, 1000])
    func largeAndSmallMagnitudes(exponent: Int) {
        // hypot(λx, λy) = λ hypot(x, y), where the squares of λx and λy overflow or
        // underflow. Scaling by a power of two is exact.
        let scale = Double(sign: .plus, exponent: exponent, significand: 1)
        for (a, b) in points {
            let x = Taylor.variable(scale * a, direction: scale, order: 3)
            let result = hypot(x, Taylor<Double>(scale * b)).coefficients.map { $0 / scale }
            expectClose(result, expected(a, b))
        }
    }

    @Test func pathsFarFromTheirDirection() {
        let large = hypot(Taylor.variable(1e200, order: 2), Taylor<Double>(1))
        #expect(large.value == 1e200)
        expectClose(Array(large.coefficients.dropFirst()), [1, 0])

        let small = hypot(Taylor.variable(1e-200, order: 2), Taylor<Double>(0))
        #expect(small.value == 1e-200)
        expectClose(Array(small.coefficients.dropFirst()), [1, 0])

        // (1e300 + t, 1e-300 + t): f' ≈ 1, f'' = (2 − f'²) / f ≈ 1 / 1e300
        let mixed = hypot(Taylor.variable(1e300, order: 2), Taylor.variable(1e-300, order: 2))
        #expect(mixed.value == 1e300)
        expectClose(mixed[1], 1)
        expectClose(mixed[2] * 1e300, 0.5)
    }

    @Test func floatScalars() {
        // The square of 1e30 overflows `Float`.
        let result = hypot(Taylor<Float>.variable(1e30, order: 2), Taylor<Float>(1))
        #expect(result.value == 1e30)
        #expect((result[1] - 1).magnitude < 1e-6)
        #expect(result[2].magnitude < 1e-6)
    }

    @Test func origin() {
        let t = Taylor.variable(0.0, order: 3)
        // Norms of lines through the origin, for t > 0.
        expectClose(hypot(t, t).coefficients, [0, 2.0.squareRoot(), 0, 0])
        expectClose(hypot(3 * t, -4 * t).coefficients, [0, 5, 0, 0])
        expectClose(hypot(-t, Taylor<Double>(0)).coefficients, [0, 1, 0, 0])
        expectClose(hypot(Taylor<Double>(0), t).coefficients, [0, 1, 0, 0])
        // hypot(t, 2t²) = t √(1 + 4t²) = t + 2t³ + …
        expectClose(hypot(t, 2 * t * t).coefficients, [0, 1, 0, 2])
        // hypot(t², −t²) = √2 t², which is smooth.
        let x = Taylor.variable(0.0, order: 4)
        expectClose(hypot(x * x, -(x * x)).coefficients, [0, 0, 2.0.squareRoot(), 0, 0])
    }

    @Test func originThroughTheDerivativeAPI() {
        // hypot(x, x²) = x √(1 + x²) = x + x³/2 − x⁵/8 + … for x > 0.
        let result = derivatives(of: { hypot($0, $0 * $0) }, at: 0.0, through: 4)
        expectClose(result, [0, 1, 0, 3, 0])
    }

    @Test func constantsAndConstantPaths() {
        #expect(hypot(Taylor<Double>(3), Taylor<Double>(4)).coefficients == [5])
        let zero = Taylor.variable(0.0, direction: 0, order: 2)
        #expect(hypot(zero + 3, zero - 4).coefficients == [5, 0, 0])
        #expect(hypot(zero, Taylor<Double>(0)).coefficients == [0, 0, 0])
    }

    @Test func nonFiniteCoefficients() {
        #expect(hypot(Taylor(coefficients: [1.0, .nan]), Taylor<Double>(1))[1].isNaN)
        #expect(hypot(Taylor(coefficients: [.infinity, 1.0]), Taylor<Double>(1)).value == .infinity)
        #expect(hypot(Taylor(coefficients: [.nan, 1.0]), Taylor<Double>(1)).value.isNaN)
    }
}
