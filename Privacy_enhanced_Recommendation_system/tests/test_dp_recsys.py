import tempfile
import unittest
import contextlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dp_recsys import DPMatrixFactorization, RatingData, load_movielens, train_test_split
from dp_recsys.__main__ import main


def toy_data() -> RatingData:
    users = np.repeat(np.arange(3), 3)
    items = np.tile(np.arange(3), 3)
    ratings = np.array([5., 4., 3., 4., 3., 2., 3., 2., 1.])
    return RatingData(users, items, ratings, 3, 3)


class RecommenderTests(unittest.TestCase):
    def test_loader_compacts_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ratings.data"
            path.write_text("10\t100\t4\t0\n20\t100\t5\t1\n")
            data = load_movielens(path)
        self.assertEqual((data.n_users, data.n_items), (2, 1))
        self.assertTrue(np.array_equal(data.users, [0, 1]))

    def test_split_keeps_training_coverage(self) -> None:
        train, test = train_test_split(toy_data(), .4, seed=3)
        self.assertGreater(len(test.ratings), 0)
        self.assertTrue(np.all(np.bincount(train.users, minlength=3) >= 1))
        self.assertTrue(np.all(np.bincount(train.items, minlength=3) >= 1))

    def test_all_methods_fit_and_predict(self) -> None:
        data = toy_data()
        for method in ("isgd", "psgd", "pals", "als"):
            model = DPMatrixFactorization(method=method, epsilon=50, factors=2, epochs=2, seed=7).fit(data)
            prediction = model.predict(np.array([0, 1]), np.array([1, 2]))
            self.assertTrue(np.all(np.isfinite(prediction)))
            self.assertTrue(np.isfinite(model.rmse(data)))
            if method in ("psgd", "pals"):
                self.assertLessEqual(np.linalg.norm(model.P_, axis=1).max(), model.p_max + 1e-12)
                self.assertLessEqual(np.linalg.norm(model.Q_, axis=1).max(), model.q_max + 1e-12)

    def test_seed_reproduces_private_output(self) -> None:
        first = DPMatrixFactorization(method="psgd", epsilon=10, factors=2, epochs=2, seed=9).fit(toy_data())
        second = DPMatrixFactorization(method="psgd", epsilon=10, factors=2, epochs=2, seed=9).fit(toy_data())
        self.assertTrue(np.allclose(first.P_, second.P_))

    def test_cli_emits_evaluation_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "u.data"
            path.write_text("1 1 5\n1 2 4\n2 1 4\n2 2 3\n3 1 3\n3 2 2\n")
            output = io.StringIO()
            with patch("sys.argv", ["dp_recsys", "--data", str(path), "--method", "als",
                                    "--factors", "2", "--epochs", "1", "--seed", "1"]):
                with contextlib.redirect_stdout(output):
                    main()
        result = json.loads(output.getvalue())
        self.assertEqual(result["method"], "als")
        self.assertGreater(result["test_ratings"], 0)
        self.assertTrue(np.isfinite(result["rmse"]))


if __name__ == "__main__":
    unittest.main()
