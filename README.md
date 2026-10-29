# Privacy-enhanced matrix factorization

This repository contains a runnable NumPy implementation of the three mechanisms evaluated in Berlioz et al., *Applying Differential Privacy to Matrix Factorization* (RecSys 2015), configured for MovieLens 100K:

- `isgd`: input perturbation followed by ordinary SGD (20 epochs by default).
- `psgd`: perturb the per-rating SGD error (5 epochs by default).
- `pals`: alternating least squares with spherical output perturbation (5 epochs by default).
- `als`: a non-private ALS reference baseline.

The previous notebooks and Streamlit experiment are retained under `Privacy_enhanced_Recommendation_system/`; the supported, reproducible interface is the `dp_recsys` package in that directory.

## Run MovieLens 100K

Download and unpack [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/), then run from this directory:

```bash
cd Privacy_enhanced_Recommendation_system
python -m dp_recsys --data ../../ml-100k/u.data --method isgd --epsilon 5 --seed 42
```

The command makes a deterministic 80/20 record split while retaining a training rating for every user and item, trains the requested model, and prints JSON containing test RMSE. The loader accepts the usual four-column `u.data` and also files whose first three whitespace-separated columns are user id, item id, rating.

For the paper's ML-100K defaults, use three factors and regularization `.06` (already the defaults). Other examples:

```bash
python -m dp_recsys --data /path/to/u.data --method psgd --epsilon 10 --epochs 5
python -m dp_recsys --data /path/to/u.data --method pals --epsilon 10 --epochs 5
python -m dp_recsys --data /path/to/u.data --method als --epsilon 1 --epochs 20
python -m unittest discover -s tests -v
```

## What is implemented

All private mechanisms first use Algorithms 1–2 from the paper: noisy global/item effects, noisy residual global/user effects, and residual clipping to `[-1, 1]`. The paper's budget allocation is preserved: `0.01ε` for each global average, `0.14ε` for each item/user effect, and `0.7ε` for factorization. Item and user shrinkage default to 10. The rating range sensitivity during preprocessing uses the fixed public bounds (1 to 5 by default); MF residual sensitivity is `2B` (`B=1`).

`psgd` adds `Laplace(k * 2B / (0.7ε))` noise to each error and clips it to `[-2, 2]`; latent rows are projected to `p_max=.4` and `q_max=.5`. `pals` samples the multivariate/spherical Laplace distribution using a Gamma radius and projects after each noisy ALS solution. Its scales follow the derivation below equation 7: user sensitivity uses `q_max`, and item sensitivity uses `p_max`. This corrects the apparent swapped labels in the printed pseudocode.

## Privacy scope and limitations

The code is intended as a faithful educational/research implementation, not a production privacy service. The stated mechanism follows the paper's **fixed observed-rating pattern, replacement adjacency** model: neighboring training sets differ in one bounded rating. Under those assumptions, with parameters selected independently of protected data, the three private modes compose to the supplied epsilon using the paper's basic composition accounting.

It does **not** claim privacy for user-level adjacency, hiding whether a user/item/rating exists, hyperparameter selection on private validation data, multiple released runs, recommendations served interactively without further accounting, or external side information. The random train/test split and reported RMSE are public evaluation steps; do not publish private evaluation results without a separate privacy analysis. The fixed default seed is for reproducibility and must not be used for a real privacy release; NumPy’s random generator is not designed for cryptographic privacy. Results will vary with epsilon, seed, split, NumPy version, and the paper does not specify every optimization convention.










 

## Local MovieLens 100K smoke run

With the dataset unpacked alongside `project`, the command above uses `../../ml-100k/u.data`. A single 80/20 split with seed 42 and epsilon 5 gave test RMSE: ISGD 0.9544, PSGD 0.9548, PALS 0.9585, and non-private ALS 0.9152. These are single-run checks, not the paper’s 10-fold experimental results.
