import Foundation

/// Pure decision logic for the short-segment re-embed + relabel post-pass.
///
/// VBx frame-level clustering assigns short speaker turns (interjections of ~0.5–2 s)
/// the label of the surrounding dominant speaker even when segmentation cuts their
/// boundaries correctly. Given a fresh embedding extracted over the segment's exact
/// audio span, this type decides — deterministically, with no model or RNG access —
/// whether the segment should move to a different speaker centroid.
///
/// All CoreML-dependent extraction lives in `OfflineEmbeddingExtractor.embedSpan`;
/// keeping the decision pure makes it unit-testable without models.
enum ShortSegmentRelabeler {

    /// Outcome of a relabel evaluation that decided to move the segment.
    struct Decision: Equatable {
        /// Zero-based cluster index of the winning centroid.
        let newCluster: Int
        /// Cosine similarity between the span embedding and the winning centroid.
        let bestCosine: Double
        /// Cosine similarity between the span embedding and the segment's current centroid.
        let currentCosine: Double

        /// How decisively the winner beat the current label.
        var margin: Double { bestCosine - currentCosine }
    }

    /// Decide whether a re-extracted span embedding should move a segment to a
    /// different speaker centroid.
    ///
    /// - Parameters:
    ///   - embedding: Raw (not necessarily normalized) embedding over the segment's span.
    ///   - centroids: Active speaker centroids, indexed by cluster (iteration order is
    ///     by index, so results are stable across runs).
    ///   - currentCluster: Zero-based cluster index the segment currently carries.
    ///   - minCosineMargin: Minimum `(bestCosine − currentCosine)` required to relabel.
    /// - Returns: A `Decision` when the segment should be relabeled, `nil` to keep the
    ///   original label. Non-finite embeddings, empty/mismatched centroids, and
    ///   out-of-range `currentCluster` all return `nil` (keep original, never throw).
    static func decision(
        embedding: [Double],
        centroids: [[Double]],
        currentCluster: Int,
        minCosineMargin: Double
    ) -> Decision? {
        guard !embedding.isEmpty, centroids.indices.contains(currentCluster) else {
            return nil
        }
        guard embedding.allSatisfy({ $0.isFinite }) else {
            return nil
        }

        var bestIndex = -1
        var bestCosine = -Double.infinity
        var currentCosine = -Double.infinity

        for (index, centroid) in centroids.enumerated() {
            guard centroid.count == embedding.count else { return nil }
            let cosine = cosineSimilarity(embedding, centroid)
            guard cosine.isFinite else { return nil }
            if cosine > bestCosine {
                bestCosine = cosine
                bestIndex = index
            }
            if index == currentCluster {
                currentCosine = cosine
            }
        }

        guard bestIndex >= 0, bestIndex != currentCluster else {
            return nil
        }
        guard bestCosine - currentCosine >= minCosineMargin else {
            return nil
        }

        return Decision(
            newCluster: bestIndex,
            bestCosine: bestCosine,
            currentCosine: currentCosine
        )
    }

    /// Parse a `"S\(cluster + 1)"` speaker id back to its zero-based cluster index.
    /// Returns `nil` for ids that don't follow the offline pipeline's convention.
    static func clusterIndex(fromSpeakerId speakerId: String) -> Int? {
        guard speakerId.hasPrefix("S"), let ordinal = Int(speakerId.dropFirst()), ordinal >= 1 else {
            return nil
        }
        return ordinal - 1
    }

    /// Cosine similarity of two same-length vectors. Zero-norm inputs yield 0.
    static func cosineSimilarity(_ lhs: [Double], _ rhs: [Double]) -> Double {
        guard lhs.count == rhs.count, !lhs.isEmpty else { return 0 }
        var dot = 0.0
        var lhsNorm = 0.0
        var rhsNorm = 0.0
        for index in lhs.indices {
            dot += lhs[index] * rhs[index]
            lhsNorm += lhs[index] * lhs[index]
            rhsNorm += rhs[index] * rhs[index]
        }
        let denominator = (lhsNorm * rhsNorm).squareRoot()
        guard denominator > 0 else { return 0 }
        return dot / denominator
    }
}
