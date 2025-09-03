import math
import torch
import pytest

from src.core.renderer import GaussianRenderer, RenderSettings

class DummyCamera:
    def __init__(self, width=64, height=64, fov_deg=60.0, device="cpu", dtype=torch.float32):
        self._width = width
        self._height = height
        self._FoVx = math.radians(fov_deg)
        self._FoVy = math.radians(fov_deg)
        self._WV = torch.eye(4, dtype=dtype, device=device)
    def world_view_transform(self):
        return self._WV

class DummyGaussians:
    """
    最小高斯桩：
    - get_xyz: [N,3]
    - get_opacity: [N,1]  (0..1)
    - get_features: [N,16,3]，第 0 通道为 DC 颜色
    - compute_3d_covariance: [N,3,3] = diag(sigma^2)
    """
    def __init__(self, xyz, sigmas, colors_dc, opacities, device="cpu", dtype=torch.float32):
        xyz = torch.as_tensor(xyz, dtype=dtype, device=device)           # [N,3]
        sigmas = torch.as_tensor(sigmas, dtype=dtype, device=device)     # [N,3]
        colors_dc = torch.as_tensor(colors_dc, dtype=dtype, device=device)# [N,3] in [0,1]
        opacities = torch.as_tensor(opacities, dtype=dtype, device=device).view(-1, 1)  # [N,1]

        N = xyz.shape[0]
        self._xyz = xyz
        self._sig = sigmas
        self._features = torch.zeros((N, 16, 3), dtype=dtype, device=device)
        self._features[:, 0, :] = colors_dc  # DC
        self._opacity = opacities

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        return self._opacity
    @property
    def get_features(self):
        return self._features
    @property
    def get_covariance(self):
        cov = self.compute_3d_covariance()
        return cov
    def compute_3d_covariance(self):
        return torch.diag_embed(self._sig ** 2)  # [N,3,3]

# --------- Fixtures ---------
@pytest.fixture(scope="module")
def device():
    return "cpu"  # 建议 CPU 跑单测更稳定；如需 CUDA，可切换

@pytest.fixture()
def cam(device):
    return DummyCamera(width=64, height=64, fov_deg=60.0, device=device)

@pytest.fixture()
def renderer():
    return GaussianRenderer(tile_size=16, radius_min=0.01, radius_max=50.0)

@pytest.fixture()
def settings(device):
    return RenderSettings(
        image_height=64,
        image_width=64,
        bg_color=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device),
        scale_modifier=1.0,
        debug=False,
    )
    
# --------- Tests ---------

class TestRenderer:
    def setup_method(self):
        self.device = "cpu"
        self.dtype = torch.float32
        self.H = 64
        self.W = 64
        self.cam = DummyCamera(width=self.W, height=self.H, fov_deg=60.0, device=self.device, dtype=self.dtype)
        self.renderer = GaussianRenderer(tile_size=16, radius_min=0.01, radius_max=50.0)
        self.settings = RenderSettings(
            image_height=self.H,
            image_width=self.W,
            bg_color=torch.tensor([0.0, 0.0, 0.0], dtype=self.dtype, device=self.device),
            scale_modifier=1.0,
            debug=True,
        )
    def test_shapes_and_types(self):
        """基础：输出张量形状/类型正确"""
        xyz = [[0.0, 0.0, 1.0]]
        sig = [[0.01, 0.01, 0.01]]
        color_dc = [[1.0, 1.0, 1.0]]
        op = [0.8]
        
        gs = DummyGaussians(xyz, sig, color_dc, op, device=self.device, dtype=self.dtype)
        
        out = self.renderer.render(self.cam, gs, self.settings) 
        assert out["image"].shape == (3, self.H, self.W)
        assert out["alpha"].shape == (1, self.H, self.W)
        assert out["depth"].shape == (1, self.H, self.W)
        assert out["viewspace_points"].shape[1] == 2
        assert out["visibility_filter"].dtype == torch.bool
        assert out["radii"].ndim == 1
        assert out["conics"].shape[-2:] == (2, 2)
        
    def test_culling_all_behind(self):
        """Z<0 全不可见 -> 返回背景 & alpha 全 0"""
        cam = DummyCamera(width=32, height=32, fov_deg=60.0, device=self.device, dtype=self.dtype)
        xyz = [[0.0, 0.0, -1.0], [0.0, 0.0, -2.0]]
        sig = [[0.01, 0.01, 0.01], [0.01, 0.01, 0.01]]
        colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        op = [0.5, 0.5]
        gs = DummyGaussians(xyz, sig, colors, op, device=self.device, dtype=self.dtype)

        out = self.renderer.render(cam, gs, self.settings)
        bg = self.settings.bg_color.view(3, 1, 1).to(self.device).repeat(1, self.H, self.W)
        assert torch.allclose(out["image"], bg)
        assert torch.count_nonzero(out["alpha"]) == 0
        
    def test_front_to_back_blending_center_pixel(self):
        """两个同中心的高斯，验证中心像素的颜色/alpha/深度与理论值匹配"""
        # 理论（像素中心 w≈1）：
        cx, cy = self.W // 2, self.H // 2    

        xyz = [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]   # 近(红)、远(绿)
        # x_pix = f_x / Z * x + cx, y_pix = f_y / Z * y + cy
        # where f_x = 0.5 * W / tan(fov_x/2), f_y = 0.5 * H / tan(fov_y/2), c_x = W/2, c_y = H/2
        # thus, x_pix = 32, y_pix = 32
        
        sig = [[0.01, 0.01, 0.01], [0.01, 0.01, 0.01]]
        colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        op = [0.5, 0.5]
        gs = DummyGaussians(xyz, sig, colors, op, device=self.device, dtype=self.dtype)

        out = self.renderer.render(self.cam, gs, self.settings)
        rgb = out["image"][:, cy, cx]
        a   = out["alpha"][0, cy, cx]
        d   = out["depth"][0, cy, cx]

        # A = 0.5 + (1-0.5)*0.5 = 0.75
        # C = 0.5*sigmoed(red) + 0.5*(1-0.5)*sigmoed(green) = [0.5, 0.25, 0]
        # c0 = sigmod(torch.tensor([1.0, 0.0, 0.0]))  # [0.731..., 0.5, 0.5], 
        # c1 = sigmod(torch.tensor([0.0, 1.0, 0.0]))  # [0.5, 0.731..., 0.5]
        # expected_rgb = 0.5 * c0 + 0.25 * c1    # -> [0.4905, 0.4328, 0.375]
        # D = (0.5*1 + 0.5*(1-0.5)*2)/0.75 = (0.5 + 0.5)/0.75 = 1.333...
        assert torch.allclose(a, torch.tensor(0.75, device=self.device), atol=1e-3)
        c0 = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=self.dtype)
        c1 = torch.tensor([0.0, 1.0, 0.0], device=self.device, dtype=self.dtype)
        s0, s1 = torch.sigmoid(c0), torch.sigmoid(c1)
        expected_rgb = 0.5 * s0 + 0.25 * s1
        
        assert torch.allclose(rgb, expected_rgb, atol=1e-3)
        
        assert torch.allclose(d, torch.tensor(4/3, device=self.device), atol=2e-2)
