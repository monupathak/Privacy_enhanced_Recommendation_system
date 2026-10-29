"""Private matrix-factorization mechanisms from Berlioz et al., RecSys 2015.

The privacy statement is for a fixed observed-rating pattern and replacement
adjacency (one bounded rating changes).  Evaluation data and data-dependent
hyperparameter tuning are outside that claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import RatingData


@dataclass
class DPMatrixFactorization:
    method: str = "isgd"
    epsilon: float = 5.0
    factors: int = 3
    regularization: float = 0.06
    epochs: int | None = None
    learning_rate: float = 0.01
    clamp: float = 1.0
    error_max: float = 2.0
    p_max: float = 0.4
    q_max: float = 0.5
    item_shrinkage: float = 10.0
    user_shrinkage: float = 10.0
    rating_min: float = 1.0
    rating_max: float = 5.0
    seed: int | None = 0

    def fit(self, data: RatingData) -> "DPMatrixFactorization":
        if self.method not in {"isgd", "psgd", "pals", "als"}:
            raise ValueError("method must be isgd, psgd, pals, or als")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.factors <= 0 or self.regularization <= 0:
            raise ValueError("factors and regularization must be positive")
        self.rng_ = np.random.default_rng(self.seed)
        self.n_users_, self.n_items_ = data.n_users, data.n_items
        if self.rating_min >= self.rating_max:
            raise ValueError("rating_min must be less than rating_max")
        self.rating_min_, self.rating_max_ = self.rating_min, self.rating_max
        self.delta_ = 2.0 * self.clamp
        residuals = self._preprocess(data, private=self.method != "als")
        if self.method == "isgd":
            noisy = np.clip(residuals + self.rng_.laplace(0, self.delta_ / (.7 * self.epsilon), len(residuals)),
                            -self.clamp, self.clamp)
            self._fit_sgd(data, noisy, self.epochs or 20, private=False, bounded=False)
        elif self.method == "psgd":
            self._fit_sgd(data, residuals, self.epochs or 5, private=True, bounded=True)
        elif self.method == "pals":
            self._fit_als(data, residuals, self.epochs or 5, private=True)
        else:
            self._fit_als(data, residuals, self.epochs or 20, private=False)
        return self

    def _preprocess(self, data: RatingData, private: bool) -> np.ndarray:
        """Algorithms 1--2 plus bounded residuals; consumes 0.3 epsilon."""
        # Paper allocation: .01 global + .14 effects twice; .7 remains for MF.
        e_global, e_effect = .01 * self.epsilon, .14 * self.epsilon
        ratings = np.clip(data.ratings, self.rating_min_, self.rating_max_)
        global_noise = self.rng_.laplace(0, (self.rating_max_ - self.rating_min_) / e_global) if private else 0.0
        global_mean = (ratings.sum() + global_noise) / len(ratings)
        sums = np.bincount(data.items, weights=ratings, minlength=data.n_items)
        counts = np.bincount(data.items, minlength=data.n_items)
        item_noise = self.rng_.laplace(0, (self.rating_max_ - self.rating_min_) / e_effect, data.n_items) if private else 0.0
        item = (sums + self.item_shrinkage * global_mean + item_noise) / (counts + self.item_shrinkage)
        self.item_effect_ = np.clip(item, self.rating_min_, self.rating_max_)
        item_residuals = ratings - self.item_effect_[data.items]
        residual_noise = self.rng_.laplace(0, (self.rating_max_ - self.rating_min_) / e_global) if private else 0.0
        residual_global = (item_residuals.sum() + residual_noise) / len(item_residuals)
        user_sums = np.bincount(data.users, weights=item_residuals, minlength=data.n_users)
        user_counts = np.bincount(data.users, minlength=data.n_users)
        user_noise = self.rng_.laplace(0, (self.rating_max_ - self.rating_min_) / e_effect, data.n_users) if private else 0.0
        user = (user_sums + self.user_shrinkage * residual_global + user_noise) / (user_counts + self.user_shrinkage)
        self.user_effect_ = np.clip(user, -2.0, 2.0)
        return np.clip(ratings - self.item_effect_[data.items] - self.user_effect_[data.users],
                       -self.clamp, self.clamp)

    def _initialize_factors(self, bounded: bool) -> None:
        self.P_ = self.rng_.normal(0, .05, (self.n_users_, self.factors))
        self.Q_ = self.rng_.normal(0, .05, (self.n_items_, self.factors))
        if bounded:
            self._project(self.P_, self.p_max)
            self._project(self.Q_, self.q_max)

    @staticmethod
    def _project(matrix: np.ndarray, maximum: float) -> None:
        norms = np.linalg.norm(matrix, axis=1)
        rows = norms > maximum
        matrix[rows] *= (maximum / norms[rows])[:, None]

    def _fit_sgd(self, data: RatingData, residuals: np.ndarray, epochs: int,
                 private: bool, bounded: bool) -> None:
        self._initialize_factors(bounded)
        scale = epochs * self.delta_ / (.7 * self.epsilon) if private else 0.0
        for _ in range(epochs):
            for index in self.rng_.permutation(len(residuals)):
                u, i = data.users[index], data.items[index]
                p, q = self.P_[u].copy(), self.Q_[i].copy()
                error = residuals[index] - p @ q
                if private:
                    error = np.clip(error + self.rng_.laplace(0, scale), -self.error_max, self.error_max)
                self.Q_[i] += self.learning_rate * (error * p - self.regularization * q)
                self.P_[u] += self.learning_rate * (error * q - self.regularization * p)
                if bounded:
                    self._project(self.P_[u:u + 1], self.p_max)
                    self._project(self.Q_[i:i + 1], self.q_max)

    def _spherical_laplace(self, sensitivity: float, epsilon_step: float) -> np.ndarray:
        """Sample density proportional to exp(-epsilon_step*||b||/sensitivity)."""
        direction = self.rng_.normal(size=self.factors)
        direction /= np.linalg.norm(direction)
        radius = self.rng_.gamma(shape=self.factors, scale=sensitivity / epsilon_step)
        return direction * radius

    def _fit_als(self, data: RatingData, residuals: np.ndarray, epochs: int, private: bool) -> None:
        self._initialize_factors(private)
        by_user = [np.flatnonzero(data.users == u) for u in range(data.n_users)]
        by_item = [np.flatnonzero(data.items == i) for i in range(data.n_items)]
        identity = np.eye(self.factors)
        epsilon_step = .7 * self.epsilon / (2 * epochs)
        for _ in range(epochs):
            for u, rows in enumerate(by_user):
                items = data.items[rows]
                x = self.Q_[items]
                solution = np.linalg.solve(x.T @ x + len(rows) * self.regularization * identity,
                                           x.T @ residuals[rows])
                if private:
                    # Derivation below Eq. 7: sensitivity is q_max*Delta_r/(n_u*lambda).
                    solution += self._spherical_laplace(self.q_max * self.delta_ /
                                                        (len(rows) * self.regularization), epsilon_step)
                    solution = self._bounded_vector(solution, self.p_max)
                self.P_[u] = solution
            for i, rows in enumerate(by_item):
                users = data.users[rows]
                x = self.P_[users]
                solution = np.linalg.solve(x.T @ x + len(rows) * self.regularization * identity,
                                           x.T @ residuals[rows])
                if private:
                    # Symmetrically, item sensitivity uses p_max.
                    solution += self._spherical_laplace(self.p_max * self.delta_ /
                                                        (len(rows) * self.regularization), epsilon_step)
                    solution = self._bounded_vector(solution, self.q_max)
                self.Q_[i] = solution

    @staticmethod
    def _bounded_vector(vector: np.ndarray, maximum: float) -> np.ndarray:
        norm = np.linalg.norm(vector)
        return vector if norm <= maximum else vector * maximum / norm

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        if not hasattr(self, "P_"):
            raise RuntimeError("fit must be called before predict")
        users, items = np.asarray(users), np.asarray(items)
        if users.min() < 0 or items.min() < 0 or users.max() >= self.n_users_ or items.max() >= self.n_items_:
            raise ValueError("prediction contains an unknown user or item")
        return self.item_effect_[items] + self.user_effect_[users] + np.sum(self.P_[users] * self.Q_[items], axis=1)

    def rmse(self, data: RatingData) -> float:
        return float(np.sqrt(np.mean((self.predict(data.users, data.items) - data.ratings) ** 2)))
