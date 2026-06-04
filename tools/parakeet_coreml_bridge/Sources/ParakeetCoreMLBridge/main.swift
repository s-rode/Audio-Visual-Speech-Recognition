import CoreML
import Darwin
import FluidAudio
import Foundation

@main
enum ParakeetCoreMLBridge {
    static func main() async {
        do {
            let options = try BridgeOptions.parse(CommandLine.arguments)
            let bridge = try await Bridge(options: options)
            try writeJSON(
                BridgeResponse(
                    type: "ready",
                    model: options.model,
                    device: options.device,
                    runtime: "FluidAudio CoreML"
                )
            )
            try await bridge.run()
        } catch {
            try? writeJSON(
                BridgeResponse(
                    type: "error",
                    message: String(describing: error)
                )
            )
            exit(1)
        }
    }
}

private struct BridgeOptions {
    var model = "FluidInference/parakeet-tdt-0.6b-v3-coreml"
    var device = "ane"

    static func parse(_ arguments: [String]) throws -> BridgeOptions {
        var options = BridgeOptions()
        var index = 1
        while index < arguments.count {
            let argument = arguments[index]
            switch argument {
            case "--model":
                index += 1
                guard index < arguments.count else { throw BridgeError.missingValue("--model") }
                options.model = arguments[index]
            case "--device":
                index += 1
                guard index < arguments.count else { throw BridgeError.missingValue("--device") }
                options.device = arguments[index]
            default:
                throw BridgeError.unsupportedArgument(argument)
            }
            index += 1
        }
        return options
    }
}

private final class Bridge {
    private let options: BridgeOptions
    private let manager: AsrManager

    init(options: BridgeOptions) async throws {
        self.options = options

        let modelConfiguration = AsrModels.defaultConfiguration()
        modelConfiguration.computeUnits = try computeUnits(from: options.device)

        let models = try await AsrModels.downloadAndLoad(
            configuration: modelConfiguration,
            version: .v3,
            encoderPrecision: .int8
        )

        let manager = AsrManager(config: .default)
        try await manager.loadModels(models)
        self.manager = manager
    }

    func run() async throws {
        while let line = readLine(strippingNewline: true) {
            if line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                continue
            }

            let request: BridgeRequest
            do {
                request = try JSONDecoder().decode(BridgeRequest.self, from: Data(line.utf8))
            } catch {
                try writeJSON(BridgeResponse(type: "error", message: "invalid request JSON: \(error)"))
                continue
            }

            if request.type == "shutdown" {
                return
            }

            guard let audioPath = request.audio_path, !audioPath.isEmpty else {
                try writeJSON(BridgeResponse(type: "error", message: "missing audio_path"))
                continue
            }

            do {
                let result = try await transcribe(audioPath: audioPath, languageCode: request.language)
                try writeJSON(
                    BridgeResponse(
                        type: "transcript",
                        text: result.text,
                        confidence: Double(result.confidence),
                        duration: result.duration,
                        processing_time: result.processingTime,
                        rtfx: Double(result.rtfx)
                    )
                )
            } catch {
                try writeJSON(BridgeResponse(type: "error", message: String(describing: error)))
            }
        }
    }

    private func transcribe(audioPath: String, languageCode: String?) async throws -> ASRResult {
        let url = URL(fileURLWithPath: audioPath)
        let layers = await manager.decoderLayerCount
        var decoderState = TdtDecoderState.make(decoderLayers: layers)
        return try await manager.transcribe(
            url,
            decoderState: &decoderState,
            language: language(from: languageCode)
        )
    }
}

private struct BridgeRequest: Decodable {
    let type: String?
    let audio_path: String?
    let language: String?
}

private struct BridgeResponse: Encodable {
    let type: String
    var model: String?
    var device: String?
    var runtime: String?
    var text: String?
    var confidence: Double?
    var duration: Double?
    var processing_time: Double?
    var rtfx: Double?
    var message: String?
}

private enum BridgeError: Error, CustomStringConvertible {
    case missingValue(String)
    case unsupportedArgument(String)
    case unsupportedDevice(String)

    var description: String {
        switch self {
        case .missingValue(let argument):
            return "missing value for \(argument)"
        case .unsupportedArgument(let argument):
            return "unsupported argument: \(argument)"
        case .unsupportedDevice(let device):
            return "unsupported device '\(device)'; use ane, gpu, cpu, or all"
        }
    }
}

private func computeUnits(from device: String) throws -> MLComputeUnits {
    switch device.lowercased() {
    case "ane":
        return .cpuAndNeuralEngine
    case "gpu":
        return .cpuAndGPU
    case "cpu":
        return .cpuOnly
    case "all", "auto":
        return .all
    default:
        throw BridgeError.unsupportedDevice(device)
    }
}

private func language(from languageCode: String?) -> Language? {
    guard let languageCode, !languageCode.isEmpty else {
        return .english
    }
    return Language(rawValue: languageCode)
}

private func writeJSON<T: Encodable>(_ value: T) throws {
    let data = try JSONEncoder().encode(value)
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data([0x0A]))
}
