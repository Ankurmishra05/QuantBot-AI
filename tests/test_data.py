import unittest

from quantbot_ai.data import FEATURE_COLUMNS, generate_synthetic_dataset, split_train_test


class DataTests(unittest.TestCase):
    def test_synthetic_dataset_contains_expected_features(self) -> None:
        dataset = generate_synthetic_dataset(rows=120, seed=5)

        for column in FEATURE_COLUMNS:
            self.assertIn(column, dataset.features.columns)
        self.assertIn("close_raw", dataset.features.columns)

    def test_train_test_split_returns_non_empty_partitions(self) -> None:
        dataset = generate_synthetic_dataset(rows=120, seed=5)
        train_df, test_df = split_train_test(dataset.features, train_split=0.8)

        self.assertGreater(len(train_df), 0)
        self.assertGreater(len(test_df), 0)


if __name__ == "__main__":
    unittest.main()
