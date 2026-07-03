import Foundation

/// Debug-only empirical-study instrumentation for the offline diarizer.
///
/// Gated on the `FLUID_OFFLINE_POSTERIOR_DUMP` environment variable, which must
/// contain a directory path. When set, the offline pipeline writes JSONL dumps of:
/// - raw per-window powerset binary activations (selected windows only, bounded volume)
/// - the aggregated global frame timeline built by `OfflineReconstruction`
/// - per-embedding VBx assignments and gamma
/// - raw / reconstructed / final emitted segments
///
/// When the variable is unset this is a no-op. Intended for offline analysis of
/// segmentation/clustering behaviour; not part of the production pipeline.
enum OfflinePosteriorDump {
    /// Resolved once per process. Nil when the env var is unset or empty.
    static let directory: URL? = {
        guard
            let path = ProcessInfo.processInfo.environment["FLUID_OFFLINE_POSTERIOR_DUMP"],
            !path.isEmpty
        else { return nil }
        let url = URL(fileURLWithPath: path, isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        } catch {
            return nil
        }
        return url
    }()

    static var isEnabled: Bool { directory != nil }

    /// Windows whose raw per-frame activations are dumped (seconds). Bounds volume.
    static let windowDumpRanges: [(start: Double, end: Double)] = [(58.0, 62.0), (72.0, 77.0)]

    /// Serialize each object as one JSON line and write the whole file at once.
    static func writeJSONL(_ objects: [[String: Any]], to fileName: String) {
        guard let directory else { return }
        var lines: [String] = []
        lines.reserveCapacity(objects.count)
        for object in objects {
            guard
                let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]),
                let line = String(data: data, encoding: .utf8)
            else { continue }
            lines.append(line)
        }
        let payload = lines.joined(separator: "\n") + "\n"
        try? payload.data(using: .utf8)?.write(to: directory.appendingPathComponent(fileName))
    }

    static func overlapsDumpRange(start: Double, end: Double) -> Bool {
        windowDumpRanges.contains { start < $0.end && end > $0.start }
    }

    /// Round for compact JSON output.
    static func round4(_ value: Double) -> Double {
        (value * 10_000).rounded() / 10_000
    }
}
