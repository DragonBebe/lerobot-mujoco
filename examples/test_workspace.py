#!/usr/bin/env python3
"""
工作空间限制测试 - test_workspace.py
验证新的工作空间限制是否合理
"""

import numpy as np
import math

def check_workspace_limits(target_pos):
    """测试版的工作空间检查函数"""
    x, y, z = target_pos
    
    print(f"🔍 检查: X={x*1000:.0f}mm, Y={y*1000:.0f}mm, Z={z*1000:.0f}mm")
    
    # 基于实际机械臂DH参数的限制
    max_reach = 0.43  # 430mm最大理论距离
    min_reach = 0.05  # 50mm最小距离
    
    # 计算到原点的距离
    reach = np.sqrt(x**2 + y**2 + z**2)
    
    # 检查最大距离
    if reach > max_reach:
        print(f"❌ 超出最大工作距离: {reach*1000:.0f}mm (最大: {max_reach*1000:.0f}mm)")
        return False
    
    # 检查最小距离
    if reach < min_reach:
        print(f"❌ 距离过近: {reach*1000:.0f}mm (最小: {min_reach*1000:.0f}mm)")
        return False
    
    # X轴：允许0到最大伸展距离
    if x < -0.1 or x > max_reach:
        print(f"❌ X轴超出范围: {x*1000:.0f}mm (范围: -100mm到{max_reach*1000:.0f}mm)")
        return False
        
    # Y轴：左右对称
    max_y = max_reach * 0.8  # 约344mm
    if abs(y) > max_y:
        print(f"❌ Y轴超出范围: {y*1000:.0f}mm (范围: ±{max_y*1000:.0f}mm)")
        return False
        
    # Z轴
    if z < -0.05 or z > 0.4:
        print(f"❌ Z轴超出范围: {z*1000:.0f}mm (范围: -50mm到400mm)")
        return False
    
    # 奇异点检查
    horizontal_dist = np.sqrt(x**2 + y**2)
    if horizontal_dist < 0.02 and abs(z) > 0.3:
        print(f"⚠️ 接近奇异点：水平距离{horizontal_dist*1000:.0f}mm，高度{z*1000:.0f}mm")
    
    print(f"✅ 通过 (距离: {reach*1000:.0f}mm)")
    return True

def test_positions():
    """测试各种位置"""
    print("🧪 工作空间限制测试")
    print("=" * 50)
    
    test_cases = [
        # 用户之前测试的位置
        ([0.0, -0.3, 0.3], "用户测试1: 正上方远距离"),
        ([0.0, -0.3, 0.2], "用户测试2: 正上方中距离"),
        
        # 基本位置
        ([0.0, 0.0, 0.1], "正上方100mm"),
        ([0.2, 0.0, 0.1], "前方200mm"),
        ([0.0, 0.2, 0.1], "右侧200mm"),
        ([0.0, -0.2, 0.1], "左侧200mm"),
        
        # 边界测试
        ([0.4, 0.0, 0.1], "最大X距离"),
        ([0.0, 0.34, 0.0], "最大Y距离"),
        ([0.0, 0.0, 0.39], "最大Z距离"),
        
        # 极限情况
        ([0.43, 0.0, 0.0], "理论最大距离"),
        ([0.05, 0.0, 0.0], "最小安全距离"),
        ([0.02, 0.0, 0.0], "过近距离"),
        
        # 红色cube位置
        ([0.25, 0.0, 0.05], "红色cube位置"),
        
        # 奇异点测试
        ([0.01, 0.0, 0.35], "接近奇异点"),
    ]
    
    passed = 0
    total = len(test_cases)
    
    for position, description in test_cases:
        print(f"\n📍 {description}")
        if check_workspace_limits(position):
            passed += 1
        print("-" * 30)
    
    print(f"\n📊 测试结果: {passed}/{total} 通过 ({passed/total*100:.1f}%)")
    
    if passed >= total * 0.7:  # 70%通过率算合理
        print("✅ 工作空间限制设置合理")
    else:
        print("⚠️ 工作空间限制可能过于严格")

if __name__ == "__main__":
    test_positions() 