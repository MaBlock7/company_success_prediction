import numpy as np


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the cosine similarity between a batch of vectors `a` and a vector `b`.

    Args:
        a (np.ndarray): Array of shape (n, d), where n is the number of samples and d is the embedding dimension.
        b (np.ndarray): Array of shape (d,), the reference embedding.

    Returns:
        np.ndarray: Cosine similarity scores of shape (n,).
    """
    return (a @ b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b) + 1e-9)
