import torch

def rotation_matrix_x(angles):
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    one = torch.ones_like(angles)
    zero = torch.zeros_like(angles)
    rotation_matrices = torch.stack([
        torch.stack([ one, zero,  zero], dim=-1),
        torch.stack([zero,  cos,  -sin], dim=-1),
        torch.stack([zero,  sin,   cos], dim=-1),
    ], dim=-2)
    return rotation_matrices

def expand_to_4x4(matrices):
    matrices_4x4 = torch.eye(4).to(matrices)
    matrices_4x4 = matrices_4x4.reshape(*[1] * len(matrices.shape[:-2]), 4, 4)
    matrices_4x4 = matrices_4x4.repeat(*matrices.shape[:-2], 1, 1)
    matrices_4x4[..., :matrices.shape[-2], :matrices.shape[-1]] = matrices
    return matrices_4x4