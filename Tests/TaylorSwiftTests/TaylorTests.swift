import Testing
import TaylorSwift

@Suite struct TaylorTests {
    @Test func variable() {
        let x = Taylor.variable(2.0, order: 3)
        #expect(x.coefficients == [2, 1, 0, 0])
        #expect(x.value == 2)
        #expect(x.order == 3)
        #expect(!x.isConstant)

        let y = Taylor.variable(2.0, direction: -0.5, order: 2)
        #expect(y.coefficients == [2, -0.5, 0])
    }

    @Test func variableOfOrderZero() {
        let x = Taylor.variable(2.0, order: 0)
        #expect(x.coefficients == [2])
        #expect(x.isConstant)
    }

    @Test func constant() {
        let c = Taylor<Double>(3.0)
        #expect(c.coefficients == [3])
        #expect(c.order == 0)
        #expect(c.isConstant)
        // The higher coefficients of a constant are zero.
        #expect(c[0] == 3)
        #expect(c[5] == 0)
        #expect(c.derivative(5) == 0)
    }

    @Test func literals() {
        let a: Taylor<Double> = 3
        let b: Taylor<Double> = 2.5
        #expect(a.coefficients == [3])
        #expect(b.coefficients == [2.5])
    }

    @Test func derivativesAreScaledCoefficients() {
        let x = Taylor(coefficients: [1.0, 2, 3, 4, 5])
        #expect(x.derivatives == [1, 2, 6, 24, 120])
        #expect(x[3] == 4)
        #expect(x.derivative(0) == 1)
        #expect(x.derivative(1) == 2)
        #expect(x.derivative(2) == 6)
        #expect(x.derivative(3) == 24)
        #expect(x.derivative(4) == 120)
    }

    @Test func initFromDerivatives() {
        let x = Taylor(derivatives: [1.0, 2, 6, 24, 120])
        #expect(x.coefficients == [1, 2, 3, 4, 5])
        #expect(Taylor(derivatives: [7.0]).coefficients == [7])
    }

    @Test func evaluation() {
        // 1 + 2h + 3h²
        let x = Taylor(coefficients: [1.0, 2, 3])
        #expect(x.evaluated(at: 0) == 1)
        #expect(x.evaluated(at: 2) == 17)
        #expect(Taylor<Double>(4.0).evaluated(at: 10) == 4)
    }

    @Test func equality() {
        let x = Taylor.variable(2.0, order: 2)
        #expect(x == Taylor(coefficients: [2.0, 1, 0]))
        #expect(x != Taylor(coefficients: [2.0, 1, 1]))
        #expect(x != Taylor(coefficients: [2.0, 2]))

        // A series equals its extension by zeros, in both directions.
        #expect(x == Taylor(coefficients: [2.0, 1]))
        #expect(Taylor(coefficients: [2.0, 1]) == x)
        #expect(x == Taylor(coefficients: [2.0, 1, 0, 0, 0]))
        #expect(Taylor(coefficients: [2.0, 1, 1]) != Taylor(coefficients: [2.0, 1]))
        #expect(Taylor(coefficients: [2.0, 1]) != Taylor(coefficients: [2.0, 1, 1]))

        // A constant equals a constant path of any order.
        #expect(Taylor<Double>(3.0) == Taylor(coefficients: [3.0, 0, 0]))
        #expect(Taylor(coefficients: [3.0, 0, 0]) == Taylor<Double>(3.0))
        #expect(Taylor<Double>(3.0) != Taylor(coefficients: [3.0, 0, 1]))
        #expect(Taylor<Double>(3.0) != Taylor(coefficients: [4.0, 0, 0]))
        #expect(Taylor<Double>(3.0) == 3)
    }

    @Test func equalityIsAnEquivalenceRelation() {
        let series: [Taylor<Double>] = [
            Taylor(3.0),
            Taylor(coefficients: [3.0, 0]),
            Taylor(coefficients: [3.0, 0, 0]),
            Taylor(coefficients: [3.0, 0, 0, 0, 0]),
            Taylor(coefficients: [3.0, 1]),
            Taylor(coefficients: [3.0, 1, 0]),
            Taylor(coefficients: [3.0, 1, 2]),
            Taylor(coefficients: [3.0, 0, 2]),
            Taylor(coefficients: [-0.0, 0]),
            Taylor(0.0),
        ]
        for a in series {
            #expect(a == a)
            for b in series {
                #expect((a == b) == (b == a), "\(a) and \(b)")
                for c in series where a == b && b == c {
                    #expect(a == c, "\(a), \(b) and \(c)")
                }
            }
        }
        // Grouped by the classes that should be equal.
        #expect(series[0] == series[3] && series[1] == series[2])
        #expect(series[4] == series[5] && series[4] != series[6])
        #expect(series[8] == series[9])
    }

    @Test func equalityFollowsFloatingPoint() {
        #expect(Taylor(coefficients: [0.0, -0.0]) == Taylor(coefficients: [-0.0, 0.0]))
        let nan = Taylor(coefficients: [1.0, .nan])
        #expect(nan != nan)
        #expect(Taylor(coefficients: [1.0, .nan]) != Taylor<Double>(1.0))
    }

    @Test func description() {
        #expect(Taylor(coefficients: [1.0, 2, -0.5]).description == "1.0 + 2.0 t - 0.5 t^2 + O(t^3)")
        #expect(Taylor<Double>(-3.0).description == "-3.0")
    }

    @Test func comparisonUsesTheValue() {
        let x = Taylor.variable(2.0, order: 2)
        let y = Taylor.variable(3.0, direction: -1, order: 2)
        let three = 3.0
        #expect(x < y)
        #expect(x <= y)
        #expect(y > x)
        #expect(y >= x)
        #expect(x < three)
        #expect(three > x)
        #expect(x > 0)
        #expect(x >= 2)
        #expect(x <= 2)
        #expect(0 < x)
    }
}
