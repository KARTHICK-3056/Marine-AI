"""Tests for the export/verification pipeline.

The drift comparator is the part that decides whether a compiled model is safe to
deploy, so its matching and classification logic is covered directly rather than
only via a full (multi-minute) export run.
"""

import os
import sys
import unittest
from typing import ClassVar

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optimize_model as om


def _det(cls, conf, box, name="shrimp"):
    return {"cls": cls, "name": name, "conf": conf, "xyxy": list(box)}


def _run(images):
    return {"images": images, "latency_ms": {"mean": 1.0, "median": 1.0, "min": 1.0, "max": 1.0, "count": 1}}


class TestIoU(unittest.TestCase):
    def test_identical_boxes(self):
        self.assertAlmostEqual(om.iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)

    def test_disjoint_boxes(self):
        self.assertEqual(om.iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_touching_edges_do_not_overlap(self):
        self.assertEqual(om.iou([0, 0, 10, 10], [10, 0, 20, 10]), 0.0)

    def test_half_overlap(self):
        # Two 10x10 boxes offset by 5 in x: intersection 50, union 150.
        self.assertAlmostEqual(om.iou([0, 0, 10, 10], [5, 0, 15, 10]), 50 / 150)

    def test_degenerate_box_is_not_a_division_error(self):
        self.assertEqual(om.iou([5, 5, 5, 5], [0, 0, 10, 10]), 0.0)


class TestCompareRuns(unittest.TestCase):
    TOL: ClassVar[dict[str, float]] = {"match_iou": 0.5, "iou_tol": 0.85, "conf_tol": 0.1}

    def test_identical_runs_have_no_issues(self):
        run = _run({"a.jpg": [_det(0, 0.9, (0, 0, 10, 10))]})
        result = om.compare_runs(run, run, **self.TOL)
        self.assertEqual(result["counters"]["matched"], 1)
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["clean_match_rate"], 1.0)
        self.assertAlmostEqual(result["mean_iou"], 1.0)

    def test_missing_detection_is_flagged(self):
        ref = _run({"a.jpg": [_det(0, 0.9, (0, 0, 10, 10))]})
        cand = _run({"a.jpg": []})
        result = om.compare_runs(ref, cand, **self.TOL)
        self.assertEqual(result["counters"]["missed"], 1)
        self.assertEqual(result["issues"][0]["kind"], "missed")

    def test_extra_detection_is_flagged_as_false_positive(self):
        ref = _run({"a.jpg": []})
        cand = _run({"a.jpg": [_det(0, 0.9, (0, 0, 10, 10))]})
        result = om.compare_runs(ref, cand, **self.TOL)
        self.assertEqual(result["counters"]["extra"], 1)
        self.assertEqual(result["issues"][0]["kind"], "extra")

    def test_confidence_drift_beyond_tolerance(self):
        ref = _run({"a.jpg": [_det(0, 0.90, (0, 0, 10, 10))]})
        cand = _run({"a.jpg": [_det(0, 0.70, (0, 0, 10, 10))]})
        result = om.compare_runs(ref, cand, **self.TOL)
        self.assertEqual(result["counters"]["conf_drift"], 1)
        self.assertEqual(result["counters"]["matched"], 0)
        self.assertAlmostEqual(result["max_conf_delta"], 0.20)

    def test_confidence_drift_within_tolerance_passes(self):
        ref = _run({"a.jpg": [_det(0, 0.90, (0, 0, 10, 10))]})
        cand = _run({"a.jpg": [_det(0, 0.85, (0, 0, 10, 10))]})
        result = om.compare_runs(ref, cand, **self.TOL)
        self.assertEqual(result["counters"]["matched"], 1)
        self.assertEqual(result["counters"]["conf_drift"], 0)

    def test_loose_box_between_match_iou_and_pass_iou(self):
        """Overlapping enough to be the same object, not enough to pass."""
        ref = _run({"a.jpg": [_det(0, 0.9, (0, 0, 10, 10))]})
        cand = _run({"a.jpg": [_det(0, 0.9, (2, 0, 12, 10))]})  # IoU = 0.666
        result = om.compare_runs(ref, cand, **self.TOL)
        self.assertEqual(result["counters"]["loose_box"], 1)
        self.assertEqual(result["counters"]["missed"], 0)

    def test_barely_overlapping_box_is_missed_plus_extra_not_a_match(self):
        ref = _run({"a.jpg": [_det(0, 0.9, (0, 0, 10, 10))]})
        cand = _run({"a.jpg": [_det(0, 0.9, (8, 0, 18, 10))]})  # IoU = 0.111
        result = om.compare_runs(ref, cand, **self.TOL)
        self.assertEqual(result["counters"]["missed"], 1)
        self.assertEqual(result["counters"]["extra"], 1)

    def test_class_swap_on_the_same_box(self):
        ref = _run({"a.jpg": [_det(0, 0.9, (0, 0, 10, 10), "shrimp")]})
        cand = _run({"a.jpg": [_det(3, 0.9, (0, 0, 10, 10), "Hydromedusa")]})
        result = om.compare_runs(ref, cand, **self.TOL)
        self.assertEqual(result["counters"]["class_mismatch"], 1)
        self.assertEqual(result["counters"]["matched"], 0)

    def test_each_candidate_box_matches_at_most_one_reference_box(self):
        """Two overlapping reference boxes must not both claim one candidate."""
        ref = _run({"a.jpg": [_det(0, 0.9, (0, 0, 10, 10)), _det(0, 0.8, (1, 1, 11, 11))]})
        cand = _run({"a.jpg": [_det(0, 0.9, (0, 0, 10, 10))]})
        result = om.compare_runs(ref, cand, **self.TOL)
        self.assertEqual(result["counters"]["matched"], 1)
        self.assertEqual(result["counters"]["missed"], 1)
        self.assertEqual(result["counters"]["extra"], 0)

    def test_image_absent_from_candidate_counts_every_detection_as_missed(self):
        ref = _run({"a.jpg": [_det(0, 0.9, (0, 0, 10, 10)), _det(0, 0.8, (50, 50, 60, 60))]})
        result = om.compare_runs(ref, _run({}), **self.TOL)
        self.assertEqual(result["counters"]["missed"], 2)
        self.assertEqual(result["ref_detections"], 2)

    def test_totals_are_reported(self):
        ref = _run({"a.jpg": [_det(0, 0.9, (0, 0, 10, 10))], "b.jpg": [_det(0, 0.7, (0, 0, 5, 5))]})
        cand = _run({"a.jpg": [_det(0, 0.9, (0, 0, 10, 10))], "b.jpg": []})
        result = om.compare_runs(ref, cand, **self.TOL)
        self.assertEqual(result["ref_detections"], 2)
        self.assertEqual(result["cand_detections"], 1)
        self.assertEqual(result["clean_match_rate"], 0.5)


class TestCalibrationDataset(unittest.TestCase):
    """INT8 calibration needs a real YOLO dataset tree, not just a folder."""

    def test_staged_dataset_has_images_labels_and_yaml(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "src"
            source.mkdir()
            for i in range(3):
                (source / f"img{i}.jpg").write_bytes(b"\xff\xd8\xff\xd9")  # minimal JPEG marker pair
            (source / "notes.txt").write_text("ignored")

            out = Path(tmp) / "calib"
            yaml_path = om.build_calibration_dataset(source, {0: "shrimp", 1: "larval fish"}, out)

            self.assertTrue(yaml_path.exists())
            self.assertEqual(len(list((out / "images" / "val").glob("*.jpg"))), 3)
            # Empty labels: calibration samples the input distribution only, but
            # YOLODataset still expects a label file per image.
            self.assertEqual(len(list((out / "labels" / "val").glob("*.txt"))), 3)

            text = yaml_path.read_text()
            self.assertIn("val: images/val", text)
            self.assertIn("0: shrimp", text)
            self.assertIn("1: larval fish", text)
            self.assertIn(out.as_posix(), text)

    def test_empty_source_directory_raises(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "empty"
            source.mkdir()
            with self.assertRaises(FileNotFoundError):
                om.build_calibration_dataset(source, {0: "shrimp"}, Path(tmp) / "out")


class TestManifest(unittest.TestCase):
    def test_missing_manifest_returns_empty_structure(self):
        from pathlib import Path
        from unittest.mock import patch

        with patch.object(om, "MANIFEST_PATH", Path("does-not-exist.json")):
            self.assertEqual(om.load_manifest(), {"exports": {}})

    def test_corrupt_manifest_does_not_raise(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "exports.json"
            bad.write_text("{not json")
            with patch.object(om, "MANIFEST_PATH", bad):
                self.assertEqual(om.load_manifest(), {"exports": {}})


class TestCli(unittest.TestCase):
    def test_export_and_verify_subcommands_parse(self):
        parser = om.build_parser()
        args = parser.parse_args(["export", "--formats", "ncnn", "onnx", "--imgsz", "416"])
        self.assertEqual(args.formats, ["ncnn", "onnx"])
        self.assertEqual(args.imgsz, 416)

        args = parser.parse_args(["verify", "--iou-tol", "0.9", "--conf-tol", "0.05"])
        self.assertEqual(args.iou_tol, 0.9)
        self.assertEqual(args.conf_tol, 0.05)
        self.assertEqual(args.match_iou, 0.5)

    def test_unknown_format_is_rejected(self):
        parser = om.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["export", "--formats", "tensorrt"])


if __name__ == "__main__":
    unittest.main()
