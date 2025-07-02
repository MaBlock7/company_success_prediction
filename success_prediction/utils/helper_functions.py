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
    if a.ndim == 1 and b.ndim == 1:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    elif a.ndim == 2 and b.ndim == 2:
        return (a @ b.T) / (np.linalg.norm(a, axis=1, keepdims=True) * np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    else:
        raise ValueError(f"Incompatible shapes: a.shape={a.shape}, b.shape={b.shape}")

def angular_distance_from_cosine(cos_sim: np.ndarray) -> np.ndarray:
    """
    Convert cosine similarity values to angular distance in [0, 1].

    Angular distance is defined as:
        arccos(cos_sim) / pi

    Args:
        cos_sim (np.ndarray): Cosine similarity values (can be scalar or array-like).

    Returns:
        np.ndarray: Angular distance values in [0, 1], where 0 means identical direction, 1 means opposite direction.
    """
    return np.arccos(np.clip(cos_sim, -1, 1)) / np.pi
