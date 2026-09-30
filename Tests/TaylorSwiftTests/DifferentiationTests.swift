import RealModule
import Testing
import TaylorSwift

/// The logistic function, written generically against the protocols of swift-numerics.
func logistic<T: ElementaryFunctions & AlgebraicField>(_ x: T) -> T {
    1 / (1 + T.exp(-x))
}

/// `f(x, y, z) = x² y + x sin(z) + y z³`
func scalarField(_ v: [Taylor<Double>]) -> Taylor<Double> {
    let (x, y, z) = (v[0], v[1], v[2])
    return x * x * y + x * sin(z) + y * pow(z, 3)
}

@Suite struct UnivariateDifferentiationTests {
    @Test func taylorCoefficientsOfExp() {
        let coefficients = taylorCoefficients(of: { exp($0) }, at: 0.0, order: 3)
        expectClose(coefficients, [1, 1, 1.0 / 2, 1.0 / 6])
    }

    @Test func derivativesOfSine() {
        expectClose(derivatives(of: { sin($0) }, at: 0.0, through: 3), [0, 1, 0, -1])
    }

    @Test func derivativeOfGivenOrder() {
        #expect(derivative(of: { $0 * $0 * $0 }, at: 2.0) == 12)
        #expect(derivative(of: { $0 * $0 * $0 }, at: 2.0, order: 0) == 8)
        #expect(derivative(of: { $0 * $0 * $0 }, at: 2.0, order: 1) == 12)
        #expect(derivative(of: { $0 * $0 * $0 }, at: 2.0, order: 2) == 12)
        #expect(derivative(of: { $0 * $0 * $0 }, at: 2.0, order: 3) == 6)
        #expect(derivative(of: { $0 * $0 * $0 }, at: 2.0, order: 4) == 0)
    }

    @Test func highOrderDerivatives() {
        // d^k/dx^k 1/(1 - x) = k! at x = 0
        var factorial = 1.0
        var expected = [1.0]
        for k in 1...20 {
            factorial *= Double(k)
            expected.append(factorial)
        }
        #expect(derivatives(of: { 1 / (1 - $0) }, at: 0.0, through: 20) == expected)
    }

    @Test func constantFunctions() {
        #expect(derivatives(of: { _ in 3 }, at: 1.0, through: 2) == [3, 0, 0])
        #expect(derivative(of: { _ in Taylor<Double>(3.0) }, at: 1.0, order: 2) == 0)
    }

    @Test func genericFunctions() {
        // σ(0) = 1/2, σ' = 1/4, σ'' = 0, σ''' = -1/8
        expectClose(derivatives(of: logistic, at: 0.0, through: 3), [0.5, 0.25, 0, -0.125])
    }

    @Test func controlFlow() {
        func f(_ x: Taylor<Double>) -> Taylor<Double> {
            var result = x
            for _ in 0..<3 {
                result = x > 0 ? result * x : -result
            }
            return result
        }
        #expect(derivatives(of: f, at: 2.0, through: 4) == [16, 32, 48, 48, 24])
        #expect(derivatives(of: f, at: -2.0, through: 2) == [2, -1, 0])
    }

    @Test func throwingFunctions() {
        struct Failure: Error {}
        #expect(throws: Failure.self) {
            try derivative(of: { _ in throw Failure() }, at: 1.0)
        }
    }

    @Test func floatScalars() {
        let result: [Float] = derivatives(of: { $0 * $0 }, at: 3, through: 2)
        #expect(result == [9, 6, 2])
    }
}

@Suite struct MultivariateDifferentiationTests {
    let point = [1.5, -0.7, 0.4]
    var x: Double { point[0] }
    var y: Double { point[1] }
    var z: Double { point[2] }

    var expectedGradient: [Double] {
        [2 * x * y + Double.sin(z), x * x + z * z * z, x * Double.cos(z) + 3 * y * z * z]
    }

    var expectedHessian: [[Double]] {
        let c = Double.cos(z)
        return [
            [2 * y, 2 * x, c],
            [2 * x, 0, 3 * z * z],
            [c, 3 * z * z, -x * Double.sin(z) + 6 * y * z],
        ]
    }

    @Test func gradientOfScalarField() {
        expectClose(gradient(of: scalarField, at: point), expectedGradient)
    }

    @Test func hessianOfScalarField() {
        let actual = hessian(of: scalarField, at: point)
        #expect(actual.count == 3)
        for (row, expectedRow) in zip(actual, expectedHessian) {
            expectClose(row, expectedRow)
        }
    }

    @Test func laplacianOfScalarField() {
        let expected = expectedHessian[0][0] + expectedHessian[1][1] + expectedHessian[2][2]
        expectClose(laplacian(of: scalarField, at: point), expected)
    }

    @Test func directionalDerivatives() {
        let v = [0.3, -1.2, 2.0]
        let first = zip(expectedGradient, v).map(*).reduce(0, +)
        var second = 0.0
        for i in 0..<3 {
            for j in 0..<3 {
                second += v[i] * expectedHessian[i][j] * v[j]
            }
        }
        expectClose(directionalDerivative(of: scalarField, at: point, along: v), first)
        expectClose(directionalDerivative(of: scalarField, at: point, along: v, order: 2), second)
        expectClose(
            directionalDerivative(of: scalarField, at: point, along: v, order: 0),
            x * x * y + x * Double.sin(z) + y * z * z * z)

        let coefficients = taylorCoefficients(of: scalarField, at: point, along: v, order: 2)
        expectClose(Array(coefficients.dropFirst()), [first, second / 2])
    }

    @Test func mixedPartials() {
        let f = scalarField
        expectClose(
            mixedPartial(of: f, at: point, withRespectTo: []),
            x * x * y + x * Double.sin(z) + y * z * z * z)
        for i in 0..<3 {
            expectClose(mixedPartial(of: f, at: point, withRespectTo: [i]), expectedGradient[i])
            for j in 0..<3 {
                expectClose(
                    mixedPartial(of: f, at: point, withRespectTo: [i, j]), expectedHessian[i][j])
            }
        }
        // ∂³f / ∂x² ∂y = 2, in any order of differentiation
        expectClose(mixedPartial(of: f, at: point, withRespectTo: [0, 0, 1]), 2)
        expectClose(mixedPartial(of: f, at: point, withRespectTo: [0, 1, 0]), 2)
        expectClose(mixedPartial(of: f, at: point, withRespectTo: [1, 0, 0]), 2)
        // ∂³f / ∂x ∂z² = -sin(z)
        expectClose(mixedPartial(of: f, at: point, withRespectTo: [0, 2, 2]), -Double.sin(z))
        // ∂³f / ∂z³ = -x cos(z) + 6y
        expectClose(
            mixedPartial(of: f, at: point, withRespectTo: [2, 2, 2]),
            -x * Double.cos(z) + 6 * y)
        // ∂³f / ∂x ∂y ∂z = 0
        expectClose(mixedPartial(of: f, at: point, withRespectTo: [0, 1, 2]), 0)
        // ∂⁴f / ∂y ∂z³ = 6
        expectClose(mixedPartial(of: f, at: point, withRespectTo: [1, 2, 2, 2]), 6, tolerance: 1e-10)
        // ∂⁴f / ∂x ∂z³ = -cos(z)
        expectClose(
            mixedPartial(of: f, at: point, withRespectTo: [2, 0, 2, 2]),
            -Double.cos(z),
            tolerance: 1e-10)
    }

    @Test func mixedPartialsOfHighOrder() {
        // f = exp(x + 2y): ∂^(a+b) f / ∂x^a ∂y^b = 2^b f
        let f: ([Taylor<Double>]) -> Taylor<Double> = { exp($0[0] + 2 * $0[1]) }
        let value = Double.exp(0.1 + 2 * 0.2)
        expectClose(
            mixedPartial(of: f, at: [0.1, 0.2], withRespectTo: [0, 1, 1, 0, 1, 0]),
            8 * value,
            tolerance: 1e-9)
    }

    @Test func noVariables() {
        let f: ([Taylor<Double>]) -> Taylor<Double> = { _ in 3 }
        #expect(gradient(of: f, at: []) == [])
        #expect(hessian(of: f, at: []) == [])
        #expect(laplacian(of: f, at: []) == 0)
        #expect(mixedPartial(of: f, at: [], withRespectTo: []) == 3)
    }
}
