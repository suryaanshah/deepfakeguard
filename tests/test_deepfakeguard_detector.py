import unittest

from deepfakeguard_detector import detect_deepfake_image, detect_tampered_document


class DeepfakeImageDetectorTests(unittest.TestCase):
    def test_detects_real_image_for_low_risk_signals(self):
        report = detect_deepfake_image(
            {
                "face_warp_score": 0.1,
                "lighting_inconsistency_score": 0.05,
                "frequency_artifact_score": 0.1,
                "compression_mismatch_score": 0.1,
            }
        )

        self.assertEqual(report.label, "REAL")
        self.assertLess(report.risk_score, 0.35)

    def test_detects_deepfake_for_high_risk_signals(self):
        report = detect_deepfake_image(
            {
                "face_warp_score": 0.9,
                "lighting_inconsistency_score": 0.8,
                "frequency_artifact_score": 0.9,
                "compression_mismatch_score": 0.7,
            }
        )

        self.assertEqual(report.label, "DEEPFAKE")
        self.assertGreaterEqual(report.risk_score, 0.65)


class TamperedDocumentDetectorTests(unittest.TestCase):
    def test_detects_original_document_for_low_risk_signals(self):
        report = detect_tampered_document(
            {
                "text_region_mismatch_score": 0.05,
                "font_inconsistency_score": 0.05,
                "edge_tamper_score": 0.1,
                "hologram_mismatch_score": 0.1,
            }
        )

        self.assertEqual(report.label, "ORIGINAL")
        self.assertLess(report.risk_score, 0.3)

    def test_detects_tampered_document_for_high_risk_signals(self):
        report = detect_tampered_document(
            {
                "text_region_mismatch_score": 0.85,
                "font_inconsistency_score": 0.7,
                "edge_tamper_score": 0.8,
                "hologram_mismatch_score": 0.8,
            }
        )

        self.assertEqual(report.label, "TAMPERED")
        self.assertGreaterEqual(report.risk_score, 0.6)


if __name__ == "__main__":
    unittest.main()
