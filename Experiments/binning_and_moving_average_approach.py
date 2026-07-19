#Generated Using GPT-5.6

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
)


def sigmoid(z):
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def normalize_train_cv_test(X_train, X_cv, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    std = np.where(std == 0.0, 1.0, std)

    X_train_norm = (X_train - mean) / std
    X_cv_norm = (X_cv - mean) / std
    X_test_norm = (X_test - mean) / std

    return X_train_norm, X_cv_norm, X_test_norm, mean, std


def split_data(
    X,
    y,
    test_size=0.20,
    cv_size=0.20,
    random_state=42,
):
    X_train_cv, X_test, y_train_cv, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    cv_fraction_of_remaining = cv_size / (1.0 - test_size)

    X_train, X_cv, y_train, y_cv = train_test_split(
        X_train_cv,
        y_train_cv,
        test_size=cv_fraction_of_remaining,
        stratify=y_train_cv,
        random_state=random_state,
    )

    return X_train, X_cv, X_test, y_train, y_cv, y_test


def binary_cross_entropy(y_true, probabilities):
    probabilities = np.clip(probabilities, 1e-12, 1.0 - 1e-12)

    return -np.mean(
        y_true * np.log(probabilities)
        + (1.0 - y_true) * np.log(1.0 - probabilities)
    )


def evaluate_binary_classifier(
    y_true,
    probabilities,
    threshold=0.5,
    dataset_name="Dataset",
):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0.0
    )

    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "specificity": specificity,
        "f1_score": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "roc_auc": roc_auc_score(y_true, probabilities),
        "log_loss": log_loss(y_true, probabilities),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }

    print(f"\n{dataset_name} evaluation")
    print("-" * 45)
    print(f"Accuracy            : {metrics['accuracy']:.6f}")
    print(f"Precision           : {metrics['precision']:.6f}")
    print(f"Recall              : {metrics['recall']:.6f}")
    print(f"Specificity         : {metrics['specificity']:.6f}")
    print(f"F1 score            : {metrics['f1_score']:.6f}")
    print(f"False positive rate : {metrics['false_positive_rate']:.6f}")
    print(f"False negative rate : {metrics['false_negative_rate']:.6f}")
    print(f"ROC-AUC             : {metrics['roc_auc']:.6f}")
    print(f"Log loss            : {metrics['log_loss']:.6f}")

    print("\nConfusion matrix")
    print(
        np.array(
            [
                [tn, fp],
                [fn, tp],
            ]
        )
    )

    return metrics


class BinnedMovingAverageAdditiveClassifier:
    def __init__(
        self,
        n_bins=12,
        moving_average_window=3,
        n_epochs=200,
        learning_rate=0.05,
        l2_regularization=1.0,
        min_bin_count=3,
        tolerance=1e-6,
        patience=20,
    ):
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2.")

        if moving_average_window < 1:
            raise ValueError(
                "moving_average_window must be positive."
            )

        if moving_average_window % 2 == 0:
            raise ValueError(
                "moving_average_window must be odd."
            )

        if n_epochs < 1:
            raise ValueError("n_epochs must be positive.")

        if not 0.0 < learning_rate <= 1.0:
            raise ValueError(
                "learning_rate must be in the interval (0, 1]."
            )

        if l2_regularization < 0.0:
            raise ValueError(
                "l2_regularization cannot be negative."
            )

        if min_bin_count < 1:
            raise ValueError(
                "min_bin_count must be at least 1."
            )

        self.n_bins = n_bins
        self.moving_average_window = moving_average_window
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization
        self.min_bin_count = min_bin_count
        self.tolerance = tolerance
        self.patience = patience

        self.intercept_ = None
        self.bin_edges_ = None
        self.bin_centers_ = None
        self.feature_functions_ = None
        self.loss_history_ = None
        self.n_features_in_ = None

    def _create_quantile_bins(self, feature_values):
        quantile_positions = np.linspace(
            0.0,
            1.0,
            self.n_bins + 1,
        )

        edges = np.quantile(
            feature_values,
            quantile_positions,
        )

        edges = np.unique(edges.astype(float))

        if len(edges) < 2:
            feature_value = float(feature_values[0])

            edges = np.array(
                [
                    feature_value - 0.5,
                    feature_value + 0.5,
                ],
                dtype=float,
            )

        finite_edges = edges.copy()

        edges[0] = -np.inf
        edges[-1] = np.inf

        centers = (
            finite_edges[:-1]
            + finite_edges[1:]
        ) / 2.0

        return edges, centers

    @staticmethod
    def _assign_bins(feature_values, edges):
        return np.digitize(
            feature_values,
            edges[1:-1],
            right=False,
        )

    def _moving_average(self, values, weights=None):
        radius = self.moving_average_window // 2

        padded_values = np.pad(
            values,
            pad_width=(radius, radius),
            mode="edge",
        )

        if weights is None:
            kernel = np.ones(
                self.moving_average_window,
                dtype=float,
            )

            smoothed = np.convolve(
                padded_values,
                kernel,
                mode="valid",
            )

            return smoothed / np.sum(kernel)

        padded_weights = np.pad(
            weights,
            pad_width=(radius, radius),
            mode="edge",
        )

        weighted_values = (
            padded_values
            * padded_weights
        )

        kernel = np.ones(
            self.moving_average_window,
            dtype=float,
        )

        numerator = np.convolve(
            weighted_values,
            kernel,
            mode="valid",
        )

        denominator = np.convolve(
            padded_weights,
            kernel,
            mode="valid",
        )

        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0.0,
        )

    def _fill_sparse_bins(
        self,
        values,
        counts,
    ):
        reliable_bins = (
            counts >= self.min_bin_count
        )

        if not np.any(reliable_bins):
            return np.zeros_like(values)

        positions = np.arange(
            len(values),
            dtype=float,
        )

        filled_values = values.copy()

        filled_values[~reliable_bins] = np.interp(
            positions[~reliable_bins],
            positions[reliable_bins],
            values[reliable_bins],
        )

        return filled_values

    @staticmethod
    def _center_bin_values(
        bin_values,
        bin_counts,
    ):
        total_count = np.sum(bin_counts)

        if total_count == 0:
            return bin_values

        weighted_mean = np.sum(
            bin_values * bin_counts
        ) / total_count

        return bin_values - weighted_mean

    def fit(
        self,
        X,
        y,
        X_validation=None,
        y_validation=None,
        verbose=True,
    ):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2:
            raise ValueError(
                "X must be a two-dimensional array."
            )

        if len(y) != X.shape[0]:
            raise ValueError(
                "X and y must contain the same number of samples."
            )

        if not np.all(
            np.isin(y, [0.0, 1.0])
        ):
            raise ValueError(
                "y must contain only 0 and 1."
            )

        if X_validation is not None:
            X_validation = np.asarray(
                X_validation,
                dtype=float,
            )

            y_validation = np.asarray(
                y_validation,
                dtype=float,
            ).reshape(-1)

        n_samples, n_features = X.shape

        self.n_features_in_ = n_features
        self.bin_edges_ = []
        self.bin_centers_ = []
        self.feature_functions_ = []
        self.loss_history_ = {
            "training": [],
            "validation": [],
        }

        training_bin_indices = []

        for feature_index in range(n_features):
            edges, centers = (
                self._create_quantile_bins(
                    X[:, feature_index]
                )
            )

            bin_indices = self._assign_bins(
                X[:, feature_index],
                edges,
            )

            n_feature_bins = len(edges) - 1

            self.bin_edges_.append(edges)
            self.bin_centers_.append(centers)

            self.feature_functions_.append(
                np.zeros(
                    n_feature_bins,
                    dtype=float,
                )
            )

            training_bin_indices.append(
                bin_indices
            )

        positive_rate = np.clip(
            np.mean(y),
            1e-6,
            1.0 - 1e-6,
        )

        self.intercept_ = np.log(
            positive_rate
            / (1.0 - positive_rate)
        )

        training_logits = np.full(
            n_samples,
            self.intercept_,
            dtype=float,
        )

        best_validation_loss = np.inf
        epochs_without_improvement = 0

        best_intercept = self.intercept_
        best_feature_functions = [
            function.copy()
            for function
            in self.feature_functions_
        ]

        for epoch in range(self.n_epochs):
            for feature_index in range(n_features):
                probabilities = sigmoid(
                    training_logits
                )

                residuals = y - probabilities

                curvature = (
                    probabilities
                    * (1.0 - probabilities)
                )

                bin_indices = (
                    training_bin_indices[
                        feature_index
                    ]
                )

                n_feature_bins = len(
                    self.feature_functions_[
                        feature_index
                    ]
                )

                bin_counts = np.bincount(
                    bin_indices,
                    minlength=n_feature_bins,
                ).astype(float)

                residual_sums = np.bincount(
                    bin_indices,
                    weights=residuals,
                    minlength=n_feature_bins,
                )

                curvature_sums = np.bincount(
                    bin_indices,
                    weights=curvature,
                    minlength=n_feature_bins,
                )

                partial_bin_updates = np.divide(
                    residual_sums,
                    curvature_sums
                    + self.l2_regularization,
                    out=np.zeros_like(
                        residual_sums,
                        dtype=float,
                    ),
                    where=(
                        curvature_sums
                        + self.l2_regularization
                    ) > 0.0,
                )

                partial_bin_updates = (
                    self._fill_sparse_bins(
                        partial_bin_updates,
                        bin_counts,
                    )
                )

                smoothed_updates = (
                    self._moving_average(
                        partial_bin_updates,
                        weights=bin_counts,
                    )
                )

                centered_updates = (
                    self._center_bin_values(
                        smoothed_updates,
                        bin_counts,
                    )
                )

                scaled_updates = (
                    self.learning_rate
                    * centered_updates
                )

                self.feature_functions_[
                    feature_index
                ] += scaled_updates

                training_logits += (
                    scaled_updates[
                        bin_indices
                    ]
                )

            probabilities = sigmoid(
                training_logits
            )

            training_loss = (
                binary_cross_entropy(
                    y,
                    probabilities,
                )
            )

            self.loss_history_[
                "training"
            ].append(training_loss)

            if X_validation is not None:
                validation_probabilities = (
                    self.predict_proba(
                        X_validation
                    )[:, 1]
                )

                validation_loss = (
                    binary_cross_entropy(
                        y_validation,
                        validation_probabilities,
                    )
                )

                self.loss_history_[
                    "validation"
                ].append(validation_loss)

                if (
                    validation_loss
                    < best_validation_loss
                    - self.tolerance
                ):
                    best_validation_loss = (
                        validation_loss
                    )

                    epochs_without_improvement = 0

                    best_intercept = (
                        self.intercept_
                    )

                    best_feature_functions = [
                        function.copy()
                        for function
                        in self.feature_functions_
                    ]

                else:
                    epochs_without_improvement += 1

                if verbose and (
                    epoch == 0
                    or (epoch + 1) % 10 == 0
                ):
                    print(
                        f"Epoch {epoch + 1:4d} | "
                        f"training loss: "
                        f"{training_loss:.6f} | "
                        f"validation loss: "
                        f"{validation_loss:.6f}"
                    )

                if (
                    epochs_without_improvement
                    >= self.patience
                ):
                    if verbose:
                        print(
                            "\nEarly stopping at "
                            f"epoch {epoch + 1}."
                        )

                    break

            elif verbose and (
                epoch == 0
                or (epoch + 1) % 10 == 0
            ):
                print(
                    f"Epoch {epoch + 1:4d} | "
                    f"training loss: "
                    f"{training_loss:.6f}"
                )

        if X_validation is not None:
            self.intercept_ = best_intercept
            self.feature_functions_ = [
                function.copy()
                for function
                in best_feature_functions
            ]

        return self

    def decision_function(self, X):
        if self.intercept_ is None:
            raise RuntimeError(
                "The model must be fitted before prediction."
            )

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                "X must be a two-dimensional array."
            )

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X contains a different number of features "
                "than the training data."
            )

        logits = np.full(
            X.shape[0],
            self.intercept_,
            dtype=float,
        )

        for feature_index in range(
            self.n_features_in_
        ):
            bin_indices = self._assign_bins(
                X[:, feature_index],
                self.bin_edges_[
                    feature_index
                ],
            )

            logits += (
                self.feature_functions_[
                    feature_index
                ][bin_indices]
            )

        return logits

    def predict_proba(self, X):
        positive_probabilities = sigmoid(
            self.decision_function(X)
        )

        negative_probabilities = (
            1.0 - positive_probabilities
        )

        return np.column_stack(
            (
                negative_probabilities,
                positive_probabilities,
            )
        )

    def predict(
        self,
        X,
        threshold=0.5,
    ):
        if not 0.0 < threshold < 1.0:
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        probabilities = (
            self.predict_proba(X)[:, 1]
        )

        return (
            probabilities >= threshold
        ).astype(int)

    def get_feature_function(
        self,
        feature_index,
    ):
        if self.intercept_ is None:
            raise RuntimeError(
                "The model must be fitted first."
            )

        if not (
            0
            <= feature_index
            < self.n_features_in_
        ):
            raise IndexError(
                "feature_index is out of range."
            )

        edges = self.bin_edges_[
            feature_index
        ]

        function_values = (
            self.feature_functions_[
                feature_index
            ]
        )

        return (
            edges.copy(),
            function_values.copy(),
        )

    def feature_contributions(self, X):
        if self.intercept_ is None:
            raise RuntimeError(
                "The model must be fitted first."
            )

        X = np.asarray(X, dtype=float)

        contributions = np.zeros(
            (
                X.shape[0],
                self.n_features_in_,
            ),
            dtype=float,
        )

        for feature_index in range(
            self.n_features_in_
        ):
            bin_indices = self._assign_bins(
                X[:, feature_index],
                self.bin_edges_[
                    feature_index
                ],
            )

            contributions[
                :,
                feature_index,
            ] = self.feature_functions_[
                feature_index
            ][bin_indices]

        return contributions


def choose_threshold(
    y_true,
    probabilities,
    minimum_recall=None,
):
    thresholds = np.linspace(
        0.01,
        0.99,
        99,
    )

    best_threshold = 0.5
    best_score = -np.inf

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )

        specificity_matrix = (
            confusion_matrix(
                y_true,
                predictions,
                labels=[0, 1],
            )
        )

        tn, fp, fn, tp = (
            specificity_matrix.ravel()
        )

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0.0
        )

        if (
            minimum_recall is not None
            and recall < minimum_recall
        ):
            continue

        balanced_accuracy = (
            recall + specificity
        ) / 2.0

        if balanced_accuracy > best_score:
            best_score = balanced_accuracy
            best_threshold = threshold

    return best_threshold


def plot_learning_curve(
    loss_history,
):
    epochs = np.arange(
        1,
        len(loss_history["training"]) + 1,
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        loss_history["training"],
        label="Training loss",
    )

    if len(
        loss_history["validation"]
    ) > 0:
        validation_epochs = np.arange(
            1,
            len(
                loss_history[
                    "validation"
                ]
            ) + 1,
        )

        plt.plot(
            validation_epochs,
            loss_history["validation"],
            label="Validation loss",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy")
    plt.title(
        "Binned Moving-Average Additive Model"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_feature_function(
    model,
    feature_index,
    feature_name,
):
    edges, contributions = (
        model.get_feature_function(
            feature_index
        )
    )

    finite_edges = edges.copy()

    if np.isneginf(finite_edges[0]):
        finite_edges[0] = (
            finite_edges[1] - 1.0
        )

    if np.isposinf(finite_edges[-1]):
        finite_edges[-1] = (
            finite_edges[-2] + 1.0
        )

    centers = (
        finite_edges[:-1]
        + finite_edges[1:]
    ) / 2.0

    plt.figure(figsize=(8, 5))

    plt.plot(
        centers,
        contributions,
        marker="o",
    )

    plt.axhline(
        0.0,
        linestyle="--",
    )

    plt.xlabel(feature_name)
    plt.ylabel(
        "Contribution to malignant log-odds"
    )
    plt.title(
        f"Learned feature function: "
        f"{feature_name}"
    )
    plt.tight_layout()
    plt.show()


def main():
    data = load_breast_cancer()

    X = data.data

    # The original dataset uses:
    # 0 = malignant
    # 1 = benign
    #
    # This transformation makes malignant the positive class:
    # 1 = malignant
    # 0 = benign
    y = (
        data.target == 0
    ).astype(int)

    (
        X_train,
        X_cv,
        X_test,
        y_train,
        y_cv,
        y_test,
    ) = split_data(
        X,
        y,
        test_size=0.20,
        cv_size=0.20,
        random_state=42,
    )

    (
        X_train_norm,
        X_cv_norm,
        X_test_norm,
        mean,
        std,
    ) = normalize_train_cv_test(
        X_train,
        X_cv,
        X_test,
    )

    model = (
        BinnedMovingAverageAdditiveClassifier(
            n_bins=12,
            moving_average_window=3,
            n_epochs=300,
            learning_rate=0.05,
            l2_regularization=1.0,
            min_bin_count=3,
            tolerance=1e-6,
            patience=30,
        )
    )

    model.fit(
        X_train_norm,
        y_train,
        X_validation=X_cv_norm,
        y_validation=y_cv,
        verbose=True,
    )

    cv_probabilities = (
        model.predict_proba(
            X_cv_norm
        )[:, 1]
    )

    chosen_threshold = choose_threshold(
        y_cv,
        cv_probabilities,
    )

    print(
        "\nChosen classification threshold: "
        f"{chosen_threshold:.2f}"
    )

    evaluate_binary_classifier(
        y_cv,
        cv_probabilities,
        threshold=chosen_threshold,
        dataset_name="Cross-validation set",
    )

    test_probabilities = (
        model.predict_proba(
            X_test_norm
        )[:, 1]
    )

    evaluate_binary_classifier(
        y_test,
        test_probabilities,
        threshold=chosen_threshold,
        dataset_name="Test set",
    )

    plot_learning_curve(
        model.loss_history_
    )

    feature_indices_to_plot = [
        0,
        2,
        7,
        20,
        22,
        27,
    ]

    for feature_index in (
        feature_indices_to_plot
    ):
        plot_feature_function(
            model,
            feature_index,
            data.feature_names[
                feature_index
            ],
        )


if __name__ == "__main__":
    main()