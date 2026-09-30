import Testing
import TaylorSwift

@Suite struct ArithmeticTests {
    let u = Taylor(coefficients: [1.0, 2, 3])
    let w = Taylor(coefficients: [4.0, -1, 0.5])

    @Test func additionAndSubtraction() {
        #expect((u + w).coefficients == [5, 1, 3.5])
        #expect((u - w).coefficients == [-3, 3, 2.5])
        #expect((-u).coefficients == [-1, -2, -3])

        var v = u
        v += w
        #expect(v.coefficients == [5, 1, 3.5])
        v -= w
        #expect(v.coefficients == [1, 2, 3])
        v.negate()
        #expect(v.coefficients == [-1, -2, -3])
    }

    @Test func multiplication() {
        // (1 + 2t + 3t²)(4 - t + t²/2) = 4 + 7t + 10.5t² + O(t³)
        #expect((u * w).coefficients == [4, 7, 10.5])
        #expect((w * u).coefficients == [4, 7, 10.5])

        var v = u
        v *= w
        #expect(v.coefficients == [4, 7, 10.5])
    }

    @Test func division() {
        // The quotient inverts the product.
        expectClose(((u * w) / w).coefficients, u.coefficients)
        expectClose((u / u).coefficients, [1, 0, 0])

        // 1 / (1 - t) = 1 + t + t² + …
        let x = Taylor.variable(0.0, order: 20)
        #expect((1 / (1 - x)).coefficients == [Double](repeating: 1, count: 21))

        var v = u * w
        v /= w
        expectClose(v.coefficients, u.coefficients)
    }

    @Test func reciprocal() throws {
        // 1 / (2 + t) has the coefficients (-1)^k / 2^(k+1).
        let x = Taylor.variable(2.0, order: 3)
        #expect(try #require(x.reciprocal).coefficients == [0.5, -0.25, 0.125, -0.0625])
        #expect(Taylor.variable(0.0, order: 3).reciprocal == nil)
    }

    @Test func reciprocalFollowsTheScalar() throws {
        // As for `Double`, the reciprocal is nil where it would be subnormal, and hence
        // not accurate.
        for value in [Double.greatestFiniteMagnitude, -1e308, 1e-320] {
            #expect(value.reciprocal == nil)
            #expect(Taylor.variable(value, order: 2).reciprocal == nil)
        }
        let tiny = Double.leastNormalMagnitude
        let x = Taylor.variable(tiny, order: 1)
        #expect(try #require(x.reciprocal).value == 1 / tiny)
        #expect(Taylor<Double>(-0.0).reciprocal == nil)
    }

    @Test func arithmeticWithScalars() {
        let s = 2.0
        #expect((u + s).coefficients == [3, 2, 3])
        #expect((s + u).coefficients == [3, 2, 3])
        #expect((u - s).coefficients == [-1, 2, 3])
        #expect((s - u).coefficients == [1, -2, -3])
        #expect((u * s).coefficients == [2, 4, 6])
        #expect((s * u).coefficients == [2, 4, 6])
        #expect((u / s).coefficients == [0.5, 1, 1.5])
        expectClose((s / u).coefficients, (Taylor(coefficients: [2.0, 0, 0]) / u).coefficients)

        var v = u
        v += s
        #expect(v.coefficients == [3, 2, 3])
        v -= s
        #expect(v.coefficients == [1, 2, 3])
        v *= s
        #expect(v.coefficients == [2, 4, 6])
        v /= s
        #expect(v.coefficients == [1, 2, 3])
    }

    @Test func arithmeticWithLiterals() {
        #expect((u + 2).coefficients == [3, 2, 3])
        #expect((2 + u).coefficients == [3, 2, 3])
        #expect((u - 2).coefficients == [-1, 2, 3])
        #expect((2 - u).coefficients == [1, -2, -3])
        #expect((u * 2).coefficients == [2, 4, 6])
        #expect((2 * u).coefficients == [2, 4, 6])
        #expect((u / 2).coefficients == [0.5, 1, 1.5])
        #expect((u * 0.5).coefficients == [0.5, 1, 1.5])
        #expect((1.5 + u).coefficients == [2.5, 2, 3])
    }

    @Test func arithmeticWithConstants() {
        let c = Taylor<Double>(2.0)
        #expect((u + c).coefficients == [3, 2, 3])
        #expect((c + u).coefficients == [3, 2, 3])
        #expect((u - c).coefficients == [-1, 2, 3])
        #expect((c - u).coefficients == [1, -2, -3])
        #expect((u * c).coefficients == [2, 4, 6])
        #expect((c * u).coefficients == [2, 4, 6])
        #expect((u / c).coefficients == [0.5, 1, 1.5])
        expectClose((c / u).coefficients, (Taylor(coefficients: [2.0, 0, 0]) / u).coefficients)

        // Constants stay constants.
        #expect((c + c).coefficients == [4])
        #expect((c - c).coefficients == [0])
        #expect((c * c).coefficients == [4])
        #expect((c / c).coefficients == [1])
        #expect((-c).coefficients == [-2])
    }

    @Test func zeroIsTheAdditiveIdentity() {
        #expect(u + .zero == u)
        #expect(Taylor<Double>.zero.coefficients == [0])
    }

    @Test func numericConformance() {
        #expect(Taylor<Double>(exactly: 3)?.coefficients == [3])
        #expect(Taylor<Float>(exactly: 16_777_217) == nil)
        #expect(Taylor(coefficients: [-2.0, 1]).magnitude == 2)
    }

    @Test func polynomialsAreExact() {
        // (2 + t)³ = 8 + 12t + 6t² + t³
        let x = Taylor.variable(2.0, order: 5)
        #expect((x * x * x).coefficients == [8, 12, 6, 1, 0, 0])
        #expect((x * x * x).derivatives == [8, 12, 12, 6, 0, 0])
    }

    @Test func truncationDropsHigherOrders() {
        // (1 + t)² = 1 + 2t + O(t²)
        let x = Taylor.variable(1.0, order: 1)
        #expect((x * x).coefficients == [1, 2])
    }

    @Test func floatScalars() {
        let x = Taylor<Float>.variable(2, order: 2)
        #expect((x * x + 0.5).coefficients == [4.5, 4, 1])
        #expect((1 / x).coefficients == [0.5, -0.25, 0.125])
    }

    @Test func mismatchedOrdersTrap() async {
        await #expect(processExitsWith: .failure) {
            _ = Taylor.variable(1.0, order: 2) + Taylor.variable(1.0, order: 3)
        }
        await #expect(processExitsWith: .failure) {
            _ = Taylor.variable(1.0, order: 2) * Taylor.variable(1.0, order: 3)
        }
        await #expect(processExitsWith: .failure) {
            _ = Taylor.variable(1.0, order: 2) / Taylor.variable(1.0, order: 3)
        }
    }
}
