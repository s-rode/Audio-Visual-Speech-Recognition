// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ParakeetCoreMLBridge",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(
            name: "parakeet-coreml-bridge",
            targets: ["ParakeetCoreMLBridge"]
        )
    ],
    dependencies: [
        .package(
            url: "https://github.com/FluidInference/FluidAudio.git",
            revision: "a40cc3da2ffc6360c5310495e91dfb9bf4d83f74"
        )
    ],
    targets: [
        .executableTarget(
            name: "ParakeetCoreMLBridge",
            dependencies: [
                .product(name: "FluidAudio", package: "FluidAudio")
            ]
        )
    ]
)
