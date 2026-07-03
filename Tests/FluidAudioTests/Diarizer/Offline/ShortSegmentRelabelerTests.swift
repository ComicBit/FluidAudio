import XCTest

@testable import FluidAudio

final class ShortSegmentRelabelerTests: XCTestCase {

    // Orthogonal unit centroids: cosine(embedding, centroid) is trivially predictable.
    private let centroids: [[Double]] = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]

    // MARK: - Relabel decision

    func testRelabelsToClosestCentroidWhenMarginDecisive() {
        // Embedding is almost exactly speaker 1, but segment carries speaker 0's label.
        let decision = ShortSegmentRelabeler.decision(
            embedding: [0.1, 0.99, 0.0],
            centroids: centroids,
            currentCluster: 0,
            minCosineMargin: 0.10
        )

        XCTAssertNotNil(decision)
        XCTAssertEqual(decision?.newCluster, 1)
        XCTAssertGreaterThanOrEqual(decision?.margin ?? 0, 0.10)
    }

    func testKeepsLabelWhenBestCentroidIsCurrentLabel() {
        let decision = ShortSegmentRelabeler.decision(
            embedding: [0.99, 0.1, 0.0],
            centroids: centroids,
            currentCluster: 0,
            minCosineMargin: 0.10
        )

        XCTAssertNil(decision)
    }

    func testKeepsLabelWhenMarginTooSmall() {
        // Nearly equidistant between clusters 0 and 1: margin far below 0.10.
        let decision = ShortSegmentRelabeler.decision(
            embedding: [0.70, 0.72, 0.0],
            centroids: centroids,
            currentCluster: 0,
            minCosineMargin: 0.10
        )

        XCTAssertNil(decision)
    }

    func testExactTieKeepsOriginalLabel() {
        // Perfect tie between current cluster and another — margin is 0 < any positive margin.
        let decision = ShortSegmentRelabeler.decision(
            embedding: [1, 1, 0],
            centroids: centroids,
            currentCluster: 0,
            minCosineMargin: 0.10
        )

        XCTAssertNil(decision)
    }

    func testZeroMarginAllowsRelabelOnAnyImprovement() {
        let decision = ShortSegmentRelabeler.decision(
            embedding: [0.70, 0.75, 0.0],
            centroids: centroids,
            currentCluster: 0,
            minCosineMargin: 0.0
        )

        XCTAssertEqual(decision?.newCluster, 1)
    }

    func testMagnitudeDoesNotAffectDecision() throws {
        // Cosine similarity is scale-invariant; a scaled embedding must decide identically.
        let unit = ShortSegmentRelabeler.decision(
            embedding: [0.1, 0.99, 0.0],
            centroids: centroids,
            currentCluster: 0,
            minCosineMargin: 0.10
        )
        let scaled = ShortSegmentRelabeler.decision(
            embedding: [10.0, 99.0, 0.0],
            centroids: centroids,
            currentCluster: 0,
            minCosineMargin: 0.10
        )

        let unwrappedUnit = try XCTUnwrap(unit)
        let unwrappedScaled = try XCTUnwrap(scaled)
        XCTAssertEqual(unwrappedUnit.newCluster, unwrappedScaled.newCluster)
        XCTAssertEqual(unwrappedUnit.bestCosine, unwrappedScaled.bestCosine, accuracy: 1e-9)
        XCTAssertEqual(unwrappedUnit.currentCosine, unwrappedScaled.currentCosine, accuracy: 1e-9)
    }

    // MARK: - Degenerate inputs never relabel

    func testNaNEmbeddingKeepsOriginalLabel() {
        let decision = ShortSegmentRelabeler.decision(
            embedding: [Double.nan, 0.99, 0.0],
            centroids: centroids,
            currentCluster: 0,
            minCosineMargin: 0.10
        )

        XCTAssertNil(decision)
    }

    func testInfiniteEmbeddingKeepsOriginalLabel() {
        let decision = ShortSegmentRelabeler.decision(
            embedding: [0.0, Double.infinity, 0.0],
            centroids: centroids,
            currentCluster: 0,
            minCosineMargin: 0.10
        )

        XCTAssertNil(decision)
    }

    func testZeroEmbeddingKeepsOriginalLabel() {
        let decision = ShortSegmentRelabeler.decision(
            embedding: [0, 0, 0],
            centroids: centroids,
            currentCluster: 0,
            minCosineMargin: 0.10
        )

        XCTAssertNil(decision)
    }

    func testEmptyEmbeddingKeepsOriginalLabel() {
        XCTAssertNil(
            ShortSegmentRelabeler.decision(
                embedding: [],
                centroids: centroids,
                currentCluster: 0,
                minCosineMargin: 0.10
            )
        )
    }

    func testOutOfRangeCurrentClusterKeepsOriginalLabel() {
        XCTAssertNil(
            ShortSegmentRelabeler.decision(
                embedding: [0.1, 0.99, 0.0],
                centroids: centroids,
                currentCluster: 7,
                minCosineMargin: 0.10
            )
        )
    }

    func testMismatchedCentroidDimensionKeepsOriginalLabel() {
        XCTAssertNil(
            ShortSegmentRelabeler.decision(
                embedding: [0.1, 0.99, 0.0],
                centroids: [[1, 0, 0], [0, 1]],
                currentCluster: 0,
                minCosineMargin: 0.10
            )
        )
    }

    // MARK: - Determinism

    func testDecisionIsDeterministicAcrossRepeatedCalls() {
        let embedding: [Double] = [0.2, 0.8, 0.3]
        let first = ShortSegmentRelabeler.decision(
            embedding: embedding,
            centroids: centroids,
            currentCluster: 0,
            minCosineMargin: 0.05
        )
        for _ in 0..<10 {
            let repeated = ShortSegmentRelabeler.decision(
                embedding: embedding,
                centroids: centroids,
                currentCluster: 0,
                minCosineMargin: 0.05
            )
            XCTAssertEqual(first, repeated)
        }
    }

    func testTieBetweenTwoForeignCentroidsPicksLowestIndex() {
        // Equidistant from clusters 1 and 2, currently labeled 0 → stable winner is index 1.
        let decision = ShortSegmentRelabeler.decision(
            embedding: [0.0, 0.7, 0.7],
            centroids: centroids,
            currentCluster: 0,
            minCosineMargin: 0.10
        )

        XCTAssertEqual(decision?.newCluster, 1)
    }

    // MARK: - Speaker id parsing

    func testClusterIndexParsesConventionalIds() {
        XCTAssertEqual(ShortSegmentRelabeler.clusterIndex(fromSpeakerId: "S1"), 0)
        XCTAssertEqual(ShortSegmentRelabeler.clusterIndex(fromSpeakerId: "S12"), 11)
    }

    func testClusterIndexRejectsUnconventionalIds() {
        XCTAssertNil(ShortSegmentRelabeler.clusterIndex(fromSpeakerId: "S0"))
        XCTAssertNil(ShortSegmentRelabeler.clusterIndex(fromSpeakerId: "speaker-1"))
        XCTAssertNil(ShortSegmentRelabeler.clusterIndex(fromSpeakerId: "S"))
        XCTAssertNil(ShortSegmentRelabeler.clusterIndex(fromSpeakerId: ""))
    }

    // MARK: - Config surface

    func testShortSegmentRelabelDisabledByDefault() {
        let config = OfflineDiarizerConfig.default
        XCTAssertEqual(config.shortSegmentRelabel.maxDurationSeconds, 0)
        XCTAssertEqual(config.shortSegmentRelabel.minCosineMargin, 0.10)
        XCTAssertNoThrow(try config.validate())
    }

    func testShortSegmentRelabelConfigurable() throws {
        var config = OfflineDiarizerConfig.default
        config.shortSegmentRelabel.maxDurationSeconds = 2.0
        config.shortSegmentRelabel.minCosineMargin = 0.05
        XCTAssertNoThrow(try config.validate())
        XCTAssertEqual(config.shortSegmentRelabel.maxDurationSeconds, 2.0)
        XCTAssertEqual(config.shortSegmentRelabel.minCosineMargin, 0.05)
    }

    func testShortSegmentRelabelValidationRejectsNegatives() {
        var config = OfflineDiarizerConfig.default
        config.shortSegmentRelabel.maxDurationSeconds = -1
        XCTAssertThrowsError(try config.validate())

        config.shortSegmentRelabel.maxDurationSeconds = 2.0
        config.shortSegmentRelabel.minCosineMargin = -0.1
        XCTAssertThrowsError(try config.validate())
    }
}
