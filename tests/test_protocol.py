import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from protocol import split_subjects


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(filename=f"{group}-{label}.mp4", source_group=f"id{group}", label=label)
                     for group in range(12) for label in (0, 1)]

    def test_no_group_or_clip_overlap(self):
        split = split_subjects(self.rows)
        self.assertFalse(set(split["train_groups"]) & set(split["test_groups"]))
        self.assertFalse(set(split["train"]) & set(split["test"]))
        self.assertEqual(set(split["train"] + split["test"]), {r["filename"] for r in self.rows})
        for part in ("train", "test"):
            self.assertEqual({r["label"] for r in self.rows if r["filename"] in split[part]}, {0, 1})

    def test_reproducible_independent_of_input_order(self):
        self.assertEqual(split_subjects(self.rows), split_subjects(list(reversed(self.rows))))

    def test_duplicate_quality_cannot_leak(self):
        with self.assertRaises(ValueError):
            split_subjects(self.rows + [self.rows[0]])

    def test_impossible_class_partition_is_rejected(self):
        with self.assertRaises(ValueError):
            split_subjects([dict(filename="r", source_group="a", label=0),
                            dict(filename="f", source_group="b", label=1)])

    def test_invalid_fraction_is_rejected(self):
        for fraction in (0, 1, -0.1):
            with self.assertRaises(ValueError):
                split_subjects(self.rows, test_fraction=fraction)


if __name__ == "__main__":
    unittest.main()
