import unittest

from patientmon.dataset import split_groups

RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


class SplitGroupsTest(unittest.TestCase):
    def test_every_group_assigned_once_and_no_split_empty(self):
        sizes = {f"v{i}": 10 for i in range(3)}
        result = split_groups(sizes, RATIOS, seed=1)
        self.assertEqual(set(result), set(sizes))
        self.assertEqual(set(result.values()), {"train", "val", "test"})

    def test_deterministic_for_same_seed_and_varies_with_seed(self):
        sizes = {f"v{i}": 10 + i for i in range(20)}
        self.assertEqual(split_groups(sizes, RATIOS, 5), split_groups(sizes, RATIOS, 5))
        self.assertNotEqual(split_groups(sizes, RATIOS, 5), split_groups(sizes, RATIOS, 6))

    def test_ratios_roughly_respected_by_frame_count(self):
        sizes = {f"v{i}": 20 for i in range(40)}
        result = split_groups(sizes, RATIOS, 0)
        share = sum(sizes[g] for g, s in result.items() if s == "train") / sum(sizes.values())
        self.assertAlmostEqual(share, 0.7, delta=0.1)

    def test_too_few_groups_raises(self):
        with self.assertRaises(ValueError):
            split_groups({"a": 5, "b": 5}, RATIOS)

    def test_zero_ratio_split_is_omitted(self):
        result = split_groups({f"v{i}": 5 for i in range(6)}, {"train": 0.8, "val": 0.2, "test": 0.0})
        self.assertEqual(set(result.values()), {"train", "val"})


if __name__ == "__main__":
    unittest.main()
