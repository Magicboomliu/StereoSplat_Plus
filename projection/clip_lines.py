import torch
import torch.nn as nn
import torch.nn.functional as F

def clip_lines_to_front(lines, epsilon=1e-6):
    
    # each line segmentation has 2 points:
    # one for 
    points_1, points_2 = torch.unbind(lines, dim=-2)
    # the last dimension is the Z
    depths_1, depths_2 = points_1[..., -1:], points_2[..., -1:]

    # points1 is more near the car's face
    # points2 is more near the car's bacl
    points_1, points_2 = (
        torch.where(depths_1 > depths_2, points_1, points_2),
        torch.where(depths_1 > depths_2, points_2, points_1),
    )
    # adjust the depth with the same sequence as the points
    depths_1, depths_2 = (
        torch.where(depths_1 > depths_2, depths_1, depths_2),
        torch.where(depths_1 > depths_2, depths_2, depths_1),
    )

    # if they are in the same surface, here should be very big
    # if they are in the different surface, here should be very smaller,
    # make sure not chunk into the ground.
    weights = depths_1 / torch.clamp(depths_1 - depths_2, min=epsilon)
    weights = torch.clamp(weights, max=1.0)

    points_2 = points_1 + (points_2 - points_1) * weights
    lines = torch.stack([points_1, points_2], dim=-2)

    masks = points_1[..., -1] > 0

    return lines, masks,[points_1,points_2]