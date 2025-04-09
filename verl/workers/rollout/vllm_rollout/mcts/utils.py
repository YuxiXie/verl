import torch
import numpy as np

def calculate_diversity_score(candidates):
    if candidates is None: return 0
    
    Q_values = [sample.Q for sample in candidates]
    variance = np.var(np.asarray(Q_values))
    gap = max(Q_values) - min(Q_values)
    # return gap if max(Q_values) > 0 else gap * 0.5
    
    visit_counts = [sample.N for sample in candidates]
    gap = max(visit_counts) - min(visit_counts)
    return gap

def cosine_similarity(vector_a, vector_b):
    """
    Calculate the cosine similarity between two vectors.
    """
    dot_product = torch.dot(vector_a, vector_b)
    norm_a = torch.norm(vector_a)
    norm_b = torch.norm(vector_b)
    
    return dot_product / (norm_a * norm_b)


def vector_distance(vector_a, vector_b):
    return ((vector_a - vector_b) ** 2).mean()