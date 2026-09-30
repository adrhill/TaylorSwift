import Testing

/// Expects two arrays to agree element-wise, up to a mixed absolute and relative tolerance.
///
/// An expected NaN or infinity must be matched exactly, with the same sign.
func expectClose(
    _ actual: [Double],
    _ expected: [Double],
    tolerance: Double = 1e-12,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    guard actual.count == expected.count else {
        Issue.record(
            "\(actual) has \(actual.count) elements instead of \(expected.count)",
            sourceLocation: sourceLocation)
        return
    }
    for (k, (a, e)) in zip(actual, expected).enumerated() {
        let matches =
            e.isNaN ? a.isNaN
            : e.isInfinite ? a == e
            // Written to also fail for NaN.
            : (a - e).magnitude <= tolerance * max(1, e.magnitude)
        if !matches {
            Issue.record(
                "Element \(k) is \(a) instead of \(e); got \(actual), expected \(expected)",
                sourceLocation: sourceLocation)
        }
    }
}

/// Expects two scalars to agree, up to a mixed absolute and relative tolerance.
func expectClose(
    _ actual: Double,
    _ expected: Double,
    tolerance: Double = 1e-12,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    expectClose([actual], [expected], tolerance: tolerance, sourceLocation: sourceLocation)
}
