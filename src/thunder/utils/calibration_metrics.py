import numpy as np
import torch

from ..utils.downstream_metrics import compute_metric


def compute_calibration_metrics(
    out_proba: torch.Tensor, label: torch.Tensor, compute_ci: bool = True
) -> dict:
    """
    Computing performance metrics.
    :param out: tensor of proba predictions.
    :param label: tensor of ground-truth labels.
    :param compute_ci: whether to compute confidence intervals.
    :return: dict of conformal prediction metrics.
    """

    # torch -> numpy
    out_proba = out_proba.numpy()
    label = label.numpy()

    # Computing metrics
    ece = compute_metric(
        label,
        out_proba,
        lambda y, y_proba: expected_calibration_error(y_proba, y),
        label_indices=np.arange(len(label)),
        compute_ci=compute_ci,
    )
    mce = compute_metric(
        label,
        out_proba,
        lambda y, y_proba: maximum_calibration_error(y_proba, y),
        label_indices=np.arange(len(label)),
        compute_ci=compute_ci,
    )
    sce = compute_metric(
        label,
        out_proba,
        lambda y, y_proba: static_calibration_error(y_proba, y),
        label_indices=np.arange(len(label)),
        compute_ci=compute_ci,
    )
    ace = compute_metric(
        label,
        out_proba,
        lambda y, y_proba: adaptive_calibration_error(y_proba, y),
        label_indices=np.arange(len(label)),
        compute_ci=compute_ci,
    )
    tace = compute_metric(
        label,
        out_proba,
        lambda y, y_proba: thresholded_adaptive_calibration_error(y_proba, y),
        label_indices=np.arange(len(label)),
        compute_ci=compute_ci,
    )

    return {
        "ECE": ece,
        "MCE": mce,
        "SCE": sce,
        "ACE": ace,
        "TACE": tace,
    }


def expected_calibration_error(samples, true_labels, M=5):
    """
    Compute the Expected Calibration Error (ECE) with M bins.

    :param samples: np.ndarray of shape (n_samples, n_classes) containing model predictions (probabilities)
    :param true_labels: np.ndarray of shape (n_samples,) containing ground truth labels
    :param M: Number of bins for calibration
    :return: Expected Calibration Error (ECE) as a float
    """
    # Define bin boundaries
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers, bin_uppers = bin_boundaries[:-1], bin_boundaries[1:]

    # Get confidence and predicted label
    confidences = np.max(samples, axis=1)
    predicted_labels = np.argmax(samples, axis=1)

    # Compute accuracy per sample
    accuracies = (predicted_labels == true_labels).astype(float)

    # Initialize ECE
    ece = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Get indices of samples falling into the bin
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        count_in_bin = np.sum(in_bin)

        if count_in_bin > 0:  # Avoid empty bins
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            bin_weight = count_in_bin / len(samples)

            # Compute bin-wise contribution to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * bin_weight

    return ece


def maximum_calibration_error(samples, true_labels, M=5):
    """
    Compute the Maximum Calibration Error (MCE) with M bins.

    :param samples: np.ndarray of shape (n_samples, n_classes) containing model predictions (probabilities)
    :param true_labels: np.ndarray of shape (n_samples,) containing ground truth labels
    :param M: Number of bins for calibration
    :return: Maximum Calibration Error (MCE) as a float
    """
    # Define bin boundaries
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers, bin_uppers = bin_boundaries[:-1], bin_boundaries[1:]

    # Get confidence and predicted label
    confidences = np.max(samples, axis=1)
    predicted_labels = np.argmax(samples, axis=1)

    # Compute accuracy per sample
    accuracies = (predicted_labels == true_labels).astype(float)

    # Initialize MCE
    max_calibration_error = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Get indices of samples falling into the bin
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        count_in_bin = np.sum(in_bin)

        if count_in_bin > 0:  # Avoid empty bins
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])

            # Compute bin-wise calibration error
            bin_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)

            # Update MCE if this bin has the highest error
            max_calibration_error = max(max_calibration_error, bin_error)

    return max_calibration_error


def static_calibration_error(samples, true_labels, M=5):
    """
    Compute the Static Calibration Error (SCE) with M bins.

    :param samples: np.ndarray of shape (n_samples, n_classes) containing model predictions (probabilities)
    :param true_labels: np.ndarray of shape (n_samples,) containing ground truth labels
    :param M: Number of bins for calibration
    :return: Static Calibration Error (SCE) as a float
    """
    # Define bin boundaries
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers, bin_uppers = bin_boundaries[:-1], bin_boundaries[1:]

    # Get confidence and predicted label
    confidences = np.max(samples, axis=1)
    predicted_labels = np.argmax(samples, axis=1)

    # Compute one-hot encoded accuracy
    accuracies = (predicted_labels == true_labels).astype(float)

    # Initialize SCE
    sce = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Get indices of samples falling into the bin
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        count_in_bin = np.sum(in_bin)

        if count_in_bin > 0:  # Avoid empty bins
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])

            # Compute bin-wise contribution to SCE (equal bin weighting)
            bin_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            sce += bin_error / M  # Each bin contributes equally

    return sce


def adaptive_calibration_error(samples, true_labels, M=5):
    """
    Compute the Adaptive Calibration Error (ACE) with M adaptive bins.

    :param samples: np.ndarray of shape (n_samples, n_classes) containing model predictions (probabilities)
    :param true_labels: np.ndarray of shape (n_samples,) containing ground truth labels
    :param M: Number of bins for calibration
    :return: Adaptive Calibration Error (ACE) as a float
    """
    # Get confidence and predicted label
    confidences = np.max(samples, axis=1)
    predicted_labels = np.argmax(samples, axis=1)
    accuracies = (predicted_labels == true_labels).astype(float)

    # Sort confidences and split into M equal-sized bins
    sorted_indices = np.argsort(confidences)
    bin_size = len(confidences) // M
    ace = 0.0

    for i in range(M):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < M - 1 else len(confidences)

        bin_indices = sorted_indices[start_idx:end_idx]

        if len(bin_indices) > 0:  # Avoid empty bins
            accuracy_in_bin = np.mean(accuracies[bin_indices])
            avg_confidence_in_bin = np.mean(confidences[bin_indices])

            # Compute bin-wise contribution to ACE
            ace += np.abs(avg_confidence_in_bin - accuracy_in_bin)

    # Normalize by number of bins
    ace /= M

    return ace


def thresholded_adaptive_calibration_error(
    samples, true_labels, M=5, confidence_threshold=0.01
):
    """
    Compute the Thresholded Adaptive Calibration Error (TACE) with M adaptive bins.

    :param samples: np.ndarray of shape (n_samples, n_classes) containing model predictions (probabilities)
    :param true_labels: np.ndarray of shape (n_samples,) containing ground truth labels
    :param M: Number of bins for calibration
    :param confidence_threshold: Minimum confidence threshold for a bin to be included in the calibration error
    :return: Thresholded Adaptive Calibration Error (TACE) as a float
    """
    # Get confidence and predicted label
    confidences = np.max(samples, axis=1)
    predicted_labels = np.argmax(samples, axis=1)
    accuracies = (predicted_labels == true_labels).astype(float)

    # Sort confidences and split into M equal-sized bins
    sorted_indices = np.argsort(confidences)
    bin_size = len(confidences) // M
    tace = 0.0
    valid_bins = 0

    for i in range(M):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < M - 1 else len(confidences)

        bin_indices = sorted_indices[start_idx:end_idx]

        if len(bin_indices) > 0:
            avg_confidence_in_bin = np.mean(confidences[bin_indices])

            # Only include bins where avg confidence is above the threshold
            if avg_confidence_in_bin > confidence_threshold:
                accuracy_in_bin = np.mean(accuracies[bin_indices])
                tace += np.abs(avg_confidence_in_bin - accuracy_in_bin)
                valid_bins += 1

    # Normalize by number of valid bins to avoid bias from empty bins
    if valid_bins > 0:
        tace /= valid_bins
    else:
        tace = 0.0  # If no bins meet the threshold, return 0

    return tace
