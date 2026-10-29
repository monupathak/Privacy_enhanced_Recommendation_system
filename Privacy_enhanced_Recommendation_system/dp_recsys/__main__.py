"""Command line entry point for private MovieLens matrix factorization."""

from __future__ import annotations

import argparse
import json

from .data import load_movielens, train_test_split
from .model import DPMatrixFactorization


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RecSys'15 DP matrix factorization on MovieLens ratings.")
    parser.add_argument("--data", required=True, help="MovieLens u.data-style ratings file")
    parser.add_argument("--method", choices=("isgd", "psgd", "pals", "als"), default="isgd")
    parser.add_argument("--epsilon", type=float, default=5.0)
    parser.add_argument("--factors", type=int, default=3)
    parser.add_argument("--regularization", type=float, default=.06)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=.01)
    parser.add_argument("--test-fraction", type=float, default=.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train, test = train_test_split(load_movielens(args.data), args.test_fraction, args.seed)
    model = DPMatrixFactorization(method=args.method, epsilon=args.epsilon, factors=args.factors,
                                  regularization=args.regularization, epochs=args.epochs,
                                  learning_rate=args.learning_rate, seed=args.seed).fit(train)
    print(json.dumps({"method": args.method, "epsilon": args.epsilon, "train_ratings": len(train.ratings),
                      "test_ratings": len(test.ratings), "rmse": model.rmse(test)}, indent=2))


if __name__ == "__main__":
    main()
