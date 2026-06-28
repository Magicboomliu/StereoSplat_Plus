"""
Tests for diff_gaussian_rasterization with conf support.

Run with:
    pixi run test
or after `pixi run build`:
    pytest tests/ -v
"""

import math
import pytest
import torch

# Skip all tests if CUDA is unavailable (e.g. CI without GPU)
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_raster_settings(H=64, W=64, device="cuda"):
    from diff_gaussian_rasterization import GaussianRasterizationSettings

    fovx = math.pi / 4
    fovy = math.pi / 4
    tanfovx = math.tan(fovx / 2)
    tanfovy = math.tan(fovy / 2)

    viewmatrix = torch.eye(4, device=device)
    # simple projection: shift camera back by 5 units
    viewmatrix[2, 3] = -5.0

    projmatrix = torch.zeros(4, 4, device=device)
    projmatrix[0, 0] = 1.0 / tanfovx
    projmatrix[1, 1] = 1.0 / tanfovy
    projmatrix[2, 2] = 1.0
    projmatrix[2, 3] = 1.0
    projmatrix[3, 2] = 1.0

    return GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.zeros(3, device=device),
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=0,
        campos=torch.zeros(3, device=device),
        prefiltered=False,
        debug=False,
    )


def make_gaussians(P=16, device="cuda"):
    """Create P random Gaussians at z=5 (in front of camera)."""
    torch.manual_seed(42)
    means3D = torch.zeros(P, 3, device=device)
    means3D[:, 2] = 5.0  # place in front of camera

    means2D = torch.zeros(P, 3, device=device, requires_grad=True)

    # Spherical harmonics (degree 0 = single colour coefficient per channel)
    shs = torch.rand(P, 1, 3, device=device)

    opacities = torch.ones(P, 1, device=device) * 0.8

    scales = torch.ones(P, 3, device=device) * 0.1
    rotations = torch.zeros(P, 4, device=device)
    rotations[:, 0] = 1.0  # unit quaternion

    return means3D, means2D, shs, opacities, scales, rotations


# ---------------------------------------------------------------------------
# Tests: forward pass output shapes
# ---------------------------------------------------------------------------

class TestForwardOutputShapes:
    def test_without_conf(self):
        from diff_gaussian_rasterization import GaussianRasterizer

        H, W, P = 64, 64, 16
        settings = make_raster_settings(H, W)
        rasterizer = GaussianRasterizer(settings)
        means3D, means2D, shs, opacities, scales, rotations = make_gaussians(P)

        out = rasterizer(means3D, means2D, opacities, shs=shs,
                         scales=scales, rotations=rotations)
        color, radii, depth, alpha, conf = out

        assert color.shape == (3, H, W)
        assert radii.shape == (P,)
        assert depth.shape == (1, H, W)
        assert alpha.shape == (1, H, W)
        assert conf.shape == (1, H, W)

    def test_with_conf(self):
        from diff_gaussian_rasterization import GaussianRasterizer

        H, W, P = 64, 64, 16
        settings = make_raster_settings(H, W)
        rasterizer = GaussianRasterizer(settings)
        means3D, means2D, shs, opacities, scales, rotations = make_gaussians(P)

        confs = torch.rand(P, device="cuda") * 0.5 + 0.5  # [0.5, 1.0]
        out = rasterizer(means3D, means2D, opacities, shs=shs,
                         scales=scales, rotations=rotations, confs=confs)
        color, radii, depth, alpha, conf = out

        assert color.shape == (3, H, W)
        assert conf.shape == (1, H, W)


# ---------------------------------------------------------------------------
# Tests: conf map value properties
# ---------------------------------------------------------------------------

class TestConfMapValues:
    def test_conf_zero_when_no_gaussians_visible(self):
        """With zero opacity, conf map should be all zeros."""
        from diff_gaussian_rasterization import GaussianRasterizer

        H, W, P = 32, 32, 8
        settings = make_raster_settings(H, W)
        rasterizer = GaussianRasterizer(settings)
        means3D, means2D, shs, opacities, scales, rotations = make_gaussians(P)

        opacities = torch.zeros(P, 1, device="cuda")
        confs = torch.ones(P, device="cuda")

        _, _, _, alpha, conf = rasterizer(means3D, means2D, opacities, shs=shs,
                                           scales=scales, rotations=rotations, confs=confs)
        assert conf.abs().max().item() < 1e-5

    def test_conf_range(self):
        """conf map values must be in [0, max_conf] for conf in [0,1]."""
        from diff_gaussian_rasterization import GaussianRasterizer

        H, W, P = 32, 32, 16
        settings = make_raster_settings(H, W)
        rasterizer = GaussianRasterizer(settings)
        means3D, means2D, shs, opacities, scales, rotations = make_gaussians(P)

        confs = torch.sigmoid(torch.randn(P, device="cuda"))  # in (0,1)

        _, _, _, _, conf_map = rasterizer(means3D, means2D, opacities, shs=shs,
                                           scales=scales, rotations=rotations, confs=confs)
        assert conf_map.min().item() >= -1e-6
        assert conf_map.max().item() <= 1.0 + 1e-6

    def test_conf_scales_with_input(self):
        """Doubling conf values should roughly double the conf map on covered pixels."""
        from diff_gaussian_rasterization import GaussianRasterizer

        H, W, P = 32, 32, 16
        settings = make_raster_settings(H, W)
        rasterizer = GaussianRasterizer(settings)
        means3D, means2D, shs, opacities, scales, rotations = make_gaussians(P)

        confs_base = torch.full((P,), 0.3, device="cuda")
        confs_double = confs_base * 2

        _, _, _, alpha, conf1 = rasterizer(means3D, means2D, opacities, shs=shs,
                                            scales=scales, rotations=rotations, confs=confs_base)
        _, _, _, _,     conf2 = rasterizer(means3D, means2D, opacities, shs=shs,
                                            scales=scales, rotations=rotations, confs=confs_double)

        # Only compare pixels that were actually covered by Gaussians
        mask = (alpha.squeeze(0) > 0.01)
        assert mask.sum() > 0, "No Gaussians were rendered"
        ratio = (conf2.squeeze(0)[mask] / (conf1.squeeze(0)[mask] + 1e-8)).mean().item()
        assert abs(ratio - 2.0) < 0.1, f"Expected ratio ~2.0, got {ratio:.3f}"

    def test_conf_independent_of_color(self):
        """Changing SH coefficients (color) should not affect conf map."""
        from diff_gaussian_rasterization import GaussianRasterizer

        H, W, P = 32, 32, 16
        settings = make_raster_settings(H, W)
        rasterizer = GaussianRasterizer(settings)
        means3D, means2D, shs, opacities, scales, rotations = make_gaussians(P)
        confs = torch.ones(P, device="cuda") * 0.5

        _, _, _, _, conf1 = rasterizer(means3D, means2D, opacities, shs=shs,
                                        scales=scales, rotations=rotations, confs=confs)
        shs2 = torch.rand_like(shs)
        _, _, _, _, conf2 = rasterizer(means3D, means2D, opacities, shs=shs2,
                                        scales=scales, rotations=rotations, confs=confs)

        assert torch.allclose(conf1, conf2, atol=1e-5), \
            "conf map should not depend on SH/color"


# ---------------------------------------------------------------------------
# Tests: backward / gradients
# ---------------------------------------------------------------------------

class TestBackward:
    def test_grad_flows_to_confs(self):
        """Loss on conf_map must produce non-zero gradients on confs."""
        from diff_gaussian_rasterization import GaussianRasterizer

        H, W, P = 32, 32, 16
        settings = make_raster_settings(H, W)
        rasterizer = GaussianRasterizer(settings)
        means3D, means2D, shs, opacities, scales, rotations = make_gaussians(P)

        confs = torch.sigmoid(torch.randn(P, device="cuda")).requires_grad_(True)

        _, _, _, _, conf_map = rasterizer(means3D, means2D, opacities, shs=shs,
                                           scales=scales, rotations=rotations, confs=confs)
        loss = conf_map.mean()
        loss.backward()

        assert confs.grad is not None
        assert confs.grad.abs().sum().item() > 0, "confs gradient is zero"

    def test_grad_not_flows_to_confs_when_loss_on_color_only(self):
        """Loss only on color should not produce conf gradients."""
        from diff_gaussian_rasterization import GaussianRasterizer

        H, W, P = 32, 32, 16
        settings = make_raster_settings(H, W)
        rasterizer = GaussianRasterizer(settings)
        means3D, means2D, shs, opacities, scales, rotations = make_gaussians(P)

        confs = torch.sigmoid(torch.randn(P, device="cuda")).requires_grad_(True)

        color, _, _, _, _ = rasterizer(means3D, means2D, opacities, shs=shs,
                                        scales=scales, rotations=rotations, confs=confs)
        loss = color.mean()
        loss.backward()

        # confs grad should be zero or None since conf_map wasn't used in loss
        if confs.grad is not None:
            assert confs.grad.abs().sum().item() < 1e-6

    def test_grad_shape(self):
        """Gradient of confs must have the same shape as confs."""
        from diff_gaussian_rasterization import GaussianRasterizer

        H, W, P = 32, 32, 16
        settings = make_raster_settings(H, W)
        rasterizer = GaussianRasterizer(settings)
        means3D, means2D, shs, opacities, scales, rotations = make_gaussians(P)

        confs = torch.rand(P, device="cuda").requires_grad_(True)

        _, _, _, _, conf_map = rasterizer(means3D, means2D, opacities, shs=shs,
                                           scales=scales, rotations=rotations, confs=confs)
        conf_map.sum().backward()

        assert confs.grad.shape == confs.shape

    def test_grad_other_params_unaffected(self):
        """Adding conf should not break gradients on existing params (opacity)."""
        from diff_gaussian_rasterization import GaussianRasterizer

        H, W, P = 32, 32, 16
        settings = make_raster_settings(H, W)
        rasterizer = GaussianRasterizer(settings)
        means3D, means2D, shs, opacities_base, scales, rotations = make_gaussians(P)

        opacities = opacities_base.clone().requires_grad_(True)
        confs = torch.rand(P, device="cuda").requires_grad_(True)

        color, _, depth, alpha, conf_map = rasterizer(
            means3D, means2D, opacities, shs=shs,
            scales=scales, rotations=rotations, confs=confs)
        loss = color.mean() + depth.mean() + conf_map.mean()
        loss.backward()

        assert opacities.grad is not None
        assert opacities.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Tests: numerical gradient check
# ---------------------------------------------------------------------------

class TestGradCheck:
    def test_gradcheck_conf(self):
        """Numerical gradient check on conf via finite differences (float32)."""
        from diff_gaussian_rasterization import GaussianRasterizer

        # Use tiny scene for gradcheck speed; float32 only (CUDA kernel constraint)
        H, W, P = 16, 16, 4
        settings = make_raster_settings(H, W)
        rasterizer = GaussianRasterizer(settings)

        torch.manual_seed(0)
        means3D = torch.zeros(P, 3, device="cuda")
        means3D[:, 2] = 5.0
        means2D = torch.zeros(P, 3, device="cuda")
        shs = torch.rand(P, 1, 3, device="cuda")
        opacities = torch.full((P, 1), 0.5, device="cuda")
        scales = torch.full((P, 3), 0.3, device="cuda")  # larger splats for better pixel coverage
        rotations = torch.zeros(P, 4, device="cuda")
        rotations[:, 0] = 1.0
        confs = (torch.rand(P, device="cuda") * 0.5 + 0.25).requires_grad_(True)

        def func(c):
            _, _, _, _, conf_map = rasterizer(means3D, means2D, opacities, shs=shs,
                                               scales=scales, rotations=rotations, confs=c)
            return conf_map

        assert torch.autograd.gradcheck(func, (confs,), eps=1e-3, atol=1e-2, rtol=1e-2), \
            "gradcheck failed for confs"
