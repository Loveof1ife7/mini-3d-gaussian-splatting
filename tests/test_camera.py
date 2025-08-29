from src.core.camera import Camera, CameraUtils
import math
import torch
import numpy as np
import pytest
class TestCamera:
    def test_projection_matrices(self):
        """测试两个投影矩阵是否一致"""
        
        # 测试参数
        test_cases = [
            {
                'name': '普通视角',
                'fovX': math.radians(60),
                'fovY': math.radians(45),
                'znear': 0.1,
                'zfar': 1000.0,
                'width': 640,
                'height': 480
            },
            {
                'name': '广角',
                'fovX': math.radians(90),
                'fovY': math.radians(67.5),
                'znear': 0.1,
                'zfar': 500.0,
                'width': 1920,
                'height': 1080
            },
            {
                'name': '窄角',
                'fovX': math.radians(30),
                'fovY': math.radians(22.5),
                'znear': 0.5,
                'zfar': 2000.0,
                'width': 800,
                'height': 600
            }
        ]
        
        print("=" * 60)
        print("投影矩阵一致性测试")
        print("=" * 60)
        
        all_passed = True
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n测试用例 {i}: {case['name']}")
            print(f"fovX: {math.degrees(case['fovX']):.1f}°, fovY: {math.degrees(case['fovY']):.1f}°")
            print(f"znear: {case['znear']}, zfar: {case['zfar']}")
            print(f"分辨率: {case['width']}x{case['height']}")
            
            # 计算两个版本的投影矩阵
            P1 = CameraUtils.build_projection_matrix_v1(
                case['znear'], case['zfar'], case['fovX'], case['fovY']
            )
            
            P2 = CameraUtils.build_projection_matrix_v2(
                case['znear'], case['zfar'], case['fovX'], case['fovY'],
                case['width'], case['height']
            )
            
            # 比较矩阵
            diff = torch.abs(P1 - P2)
            max_diff = torch.max(diff).item()
            avg_diff = torch.mean(diff).item()
            
            print(f"最大差异: {max_diff:.8f}")
            print(f"平均差异: {avg_diff:.8f}")
            
            # 检查是否相等（考虑浮点精度误差）
            tolerance = 1e-6
            if max_diff < tolerance:
                print("✅ 测试通过: 两个矩阵一致")
            else:
                print("❌ 测试失败: 矩阵不一致")
                all_passed = False
                
            # 显示矩阵摘要
            print("矩阵v1 (前3x3):")
            print(P1[:3, :3].numpy())
            print("矩阵v2 (前3x3):")
            print(P2[:3, :3].numpy())
            
            # 详细数学验证
            print("\n" + "=" * 60)
            print("数学验证")
            print("=" * 60)
            
            # 取第一个测试用例进行详细验证
            case = test_cases[0]
            fovX, fovY = case['fovX'], case['fovY']
            width, height = case['width'], case['height']
            
            # 计算理论值
            tan_half_fovX = math.tan(fovX / 2)
            tan_half_fovY = math.tan(fovY / 2)
            
            theoretical_A = 1.0 / tan_half_fovX
            theoretical_B = 1.0 / tan_half_fovY
            
            # 计算v2的实际值
            focal_x = (width / 2) / tan_half_fovX
            focal_y = (height / 2) / tan_half_fovY
            actual_A_v2 = 2 * focal_x / width
            actual_B_v2 = 2 * focal_y / height
            
            print(f"理论值 A = 1/tan(fovX/2) = {theoretical_A:.6f}")
            print(f"v2计算 A = 2*focal_x/width = {actual_A_v2:.6f}")
            print(f"理论值 B = 1/tan(fovY/2) = {theoretical_B:.6f}")
            print(f"v2计算 B = 2*focal_y/height = {actual_B_v2:.6f}")
            
            print(f"\nA 差异: {abs(theoretical_A - actual_A_v2):.8f}")
            print(f"B 差异: {abs(theoretical_B - actual_B_v2):.8f}")
            
            # 最终结论
            print("\n" + "=" * 60)
            if all_passed:
                print("🎉 所有测试通过！两个函数产生相同的投影矩阵")
                print("📝 结论: width/height参数在v2中是冗余的，可以省略")
            else:
                print("💥 测试失败！请检查实现")
            print("=" * 60)

    def test_point_projection(self):
        """测试点投影结果是否一致"""
        
        print("\n" + "=" * 60)
        print("点投影测试")
        print("=" * 60)
        
        # 测试参数
        fovX, fovY = math.radians(60), math.radians(45)
        znear, zfar = 0.1, 1000.0
        width, height = 640, 480
        
        # 创建投影矩阵
        P1 = CameraUtils.build_projection_matrix_v1(znear, zfar, fovX, fovY)
        P2 = CameraUtils.build_projection_matrix_v2(znear, zfar, fovX, fovY, width, height)
        
        # 测试点
        test_points = torch.tensor([
            [0, 0, 10, 1],      # 中心前方
            [2, 1, 5, 1],       # 右上前方
            [-1, -0.5, 20, 1],  # 左下远方
            [0, 0, 0.5, 1]      # 非常近的点
        ], dtype=torch.float32)
        
        print("测试点 (齐次坐标):")
        for i, point in enumerate(test_points):
            print(f"点{i+1}: {point.numpy()}")
        
        # 投影点
        points_proj1 = (P1 @ test_points.t()).t()
        points_proj2 = (P2 @ test_points.t()).t()
        
        # 透视除法
        points_ndc1 = points_proj1[:, :3] / points_proj1[:, 3:]
        points_ndc2 = points_proj2[:, :3] / points_proj2[:, 3:]
        
        print("\n投影结果比较:")
        print("点# | v1 NDC坐标       | v2 NDC坐标       | 差异")
        print("-" * 55)
        
        for i in range(len(test_points)):
            ndc1 = points_ndc1[i].numpy()
            ndc2 = points_ndc2[i].numpy()
            diff = np.abs(ndc1 - ndc2).max()
            
            print(f"{i+1:2} | [{ndc1[0]:6.3f}, {ndc1[1]:6.3f}, {ndc1[2]:6.3f}] | "
                f"[{ndc2[0]:6.3f}, {ndc2[1]:6.3f}, {ndc2[2]:6.3f}] | {diff:.6f}")
