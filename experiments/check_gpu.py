#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check gpu 
快速检查PyTorch 的GPU 可用性与设备状态
可在训练前运行确认GPU是否在正常使用

"""
import torch
import subprocess

def main():
    print("="*60)
    print("🔍 [GPU Check] PyTorch环境与CUDA状态")
    print("="*60)
    
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"[INFO] CUDA 可用 | 检测到 {device_count} 个 GPU")
        
        for i in range(device_count):
            print(f"\n--- GPU {i} ---")
            print("设备名称:",torch.cuda.get_device_name(i))
            print("显存总量:{:.2f} GB".format(torch.cuda.get_device_properties(i).total_memory / 1024**3))
            print("当前已分配显存: {:.2f} GB".format(torch.cuda.memory_allocated(i) / 1024**3))
            print("当前已缓存显存: {:.2f} GB".format(torch.cuda.memory_reserved(i) / 1024**3))
            
        print("\n[INFO] 当前活跃设备:",torch.cuda.current_device())
        print("[INFO] CUDA 版本:",torch.version.cuda)
        print("[INFO] PyTorch 版本:",torch.__version__)
        
    else:
        print("[WARN] ⚠️ 未检测到可用的 CUDA GPU , 当前使用 CPU")
        print("[INFO] PyTorch 版本:",torch.__version__)
        
    print("\n" + "="*60)
    print(" 💻 nvidia-smi 输出 (如果已安装 NVIDIA 驱动):")
    print("="*60)
    try:
        subprocess.run(["nvidia-smi"],check=True)
    except Exception as e:
        print(f"[WARN] 无法执行 nvidia-smi: {e}")
        

if __name__ == "__main__":
    main()