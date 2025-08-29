import math
import types
import torch
import torch.nn as nn
import numpy as np
import pytest

from src.core.gaussian_model import GaussianModel
from config.config import TrainingConfig


def _quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """q=[N,4] as [w,x,y,z] -> [N,3,3]."""
    q = torch.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    R = torch.stack([
        1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy),
        2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx),
        2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)
    ], dim=-1).view(-1, 3, 3)
    return R


def make_model(n=128, extent=1.0, device="cpu"):
    cfg = TrainingConfig()
    m = GaussianModel(cfg).to(device)
    m.create_from_random(n, extent)
    return m


class TestGaussianModel:
    def test_parameter_initialization(self):
        N = 256
        m = make_model(N)
        # shapes
        assert m.get_xyz.shape == (N, 3)
        assert m._features_dc.shape == (N, 1, 3)
        assert m._features_rest.shape == (N, 15, 3)
        assert m._scaling.shape == (N, 3)
        assert m._rotation.shape == (N, 4)
        assert m._opacity.shape == (N, 1)
        # buffers
        assert m.xyz_gradient_accum.shape == (N, 3)
        assert m.denom.shape == (N, 1)
        assert m.max_radii2D.shape[0] == N

    def test_property_access(self):
        N = 64
        m = make_model(N)

        # scaling should be positive after activation
        sigma = m.get_scaling
        assert sigma.shape == (N, 3)
        assert torch.all(sigma > 0)

        # rotation is normalized quaternions
        q = m.get_rotation
        qnorm = torch.linalg.norm(q, dim=-1)
        assert torch.allclose(qnorm, torch.ones_like(qnorm), atol=1e-5)

        # opacity is in (0,1)
        a = m.get_opacity
        assert torch.all(a > 0) and torch.all(a < 1)

        # features = DC + REST
        feats = m.get_features
        assert feats.shape == (N, 16, 3)
        cat = torch.cat([m._features_dc, m._features_rest], dim=1)
        assert torch.allclose(feats, cat)

    def test_covariance_computation(self):
        N = 32
        m = make_model(N)
        cov = m.compute_3d_covariance()        # [N,3,3]
        assert cov.shape == (N, 3, 3)

        # Recompute expected: R diag(sigma^2) R^T
        sigma = m.get_scaling
        R = _quat_to_mat(m.get_rotation)
        D = torch.diag_embed(sigma ** 2)
        cov_expected = R @ D @ R.transpose(-1, -2)

        assert torch.allclose(cov, cov_expected, atol=1e-5, rtol=1e-5)
        # PSD check: eigenvalues >= 0
        eigvals = torch.linalg.eigvalsh(cov)
        assert torch.all(eigvals > -1e-6)

    def test_densify_operations(self):
        N = 64
        extent = 1.0
        m = make_model(N, extent)
        print(m)

        # --- monkeypatch missing helpers (for your current snippet) ---
        # Use instance-bound methods so self works inside density_*.
        m.get_feature_dc = types.MethodType(lambda self: self._features_dc, m)
        m.get_feature_ac = types.MethodType(lambda self: self._features_rest, m)

        # Fix _append_points if your class still references _scaling_log.
        if not hasattr(m, "_scaling_log"):
            def _append_points_patch(self, xyz, fdc, frest, scaling_log, rot, op):
                self._xyz = nn.Parameter(torch.cat([self._xyz.data, xyz], dim=0))
                self._features_dc = nn.Parameter(torch.cat([self._features_dc.data, fdc], dim=0))
                self._features_rest = nn.Parameter(torch.cat([self._features_rest.data, frest], dim=0))
                self._scaling = nn.Parameter(torch.cat([self._scaling.data, scaling_log], dim=0))
                self._rotation = nn.Parameter(torch.cat([self._rotation.data, rot], dim=0))
                self._opacity = nn.Parameter(torch.cat([self._opacity.data, op], dim=0))
            m._append_points = types.MethodType(_append_points_patch, m)

        # Give all points a gradient so they are candidates.
        g = torch.ones_like(m._xyz) * 1.0
        m._xyz.grad = g

        # Make first k "big" and next k "small"
        k = 8
        with torch.no_grad():
            # big sigmas -> trigger split (threshold is 0.03 * extent)
            big_sigma = 0.06 * extent
            m._scaling[:k] = math.log(big_sigma)
            # small sigmas -> trigger clone (threshold < 0.01 * extent)
            small_sigma = 0.005 * extent
            m._scaling[k:2*k] = math.log(small_sigma)

        # --- split ---
        N0 = m.get_num_points()
        m.density_and_split(grad_threshold=0.5, scene_extent=extent)
        # Net +k because k originals removed, 2k added
        assert m.get_num_points() == N0 + k

        # Prepare grads again (new parameters lost .grad)
        m._xyz.grad = torch.ones_like(m._xyz) * 1.0

        # --- clone ---
        N1 = m.get_num_points()
        m.density_and_clone(grad_threshold=0.5, scene_extent=extent)
        # Expect +k clones (original small ones persist)
        assert m.get_num_points() == N1 + k
