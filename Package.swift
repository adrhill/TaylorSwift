// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "TaylorSwift",
    platforms: [.macOS(.v10_15), .iOS(.v13), .tvOS(.v13), .watchOS(.v6)],
    products: [
        .library(name: "TaylorSwift", targets: ["TaylorSwift"])
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-numerics", from: "1.0.0")
    ],
    targets: [
        .target(
            name: "TaylorSwift",
            dependencies: [
                .product(name: "RealModule", package: "swift-numerics")
            ]
        ),
        .testTarget(
            name: "TaylorSwiftTests",
            dependencies: [
                "TaylorSwift",
                .product(name: "RealModule", package: "swift-numerics"),
            ]
        ),
    ]
)
