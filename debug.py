# debug_dataset.py (V2 - 带详细错误打印)
import yaml
from dataset import LibriSpeechSpeakerDataset
import os
import torch
import torchaudio # 确保导入

print("--- 开始调试数据集 V2 ---")

# 1. 加载配置文件
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("配置文件 config.yaml 加载成功。")
except Exception as e:
    print(f"!!! 错误：加载 config.yaml 失败: {e}")
    exit()

# 2. 检查路径... (这部分不变)
dataset_path = config['data']['dataset_path']
absolute_path = os.path.abspath(dataset_path) 
print(f"配置文件中的路径: {dataset_path}")
print(f"转换后的绝对路径: {absolute_path}")
if not os.path.isdir(absolute_path):
    print(f"❌ 警告：路径 '{absolute_path}' 不存在或不是一个目录！")
    exit()
else:
    print("✅ 路径存在且是一个目录。")

# 3. 创建 Dataset 实例... (这部分不变)
try:
    print("\n--- 正在创建 Dataset 实例 (train=True)... ---")
    train_dataset = LibriSpeechSpeakerDataset(config, train=True)
except Exception as e:
    print(f"!!! 错误：创建训练集实例失败: {e}")
    exit()

# 4. 【核心改动】尝试获取一个数据样本，并打印详细错误
if len(train_dataset) > 0:
    print("\n--- 正在尝试获取第一个训练样本... ---")
    
    # 我们直接从 dataset.py 复制 __getitem__ 的逻辑到这里进行测试
    # 这样可以避免 try...except 吞掉错误
    
    file_path, speaker_id = train_dataset.file_list[0]
    print(f"正在尝试加载文件: {file_path}")
    
    try:
        # --- 模拟 __getitem__ 的内部操作 ---
        waveform, sample_rate = torchaudio.load(file_path)
        print("✅ torchaudio.load() 成功！")
        
        if sample_rate != config['audio']['sample_rate']:
            print("正在重采样...")
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=config['audio']['sample_rate'])
            waveform = resampler(waveform)
            print("✅ 重采样成功！")

        segment_length_samples = int(config['audio']['sample_rate'] * config['audio']['segment_length_ms'] / 1000)
        if waveform.shape[1] < segment_length_samples:
            print(f"❌ 警告：音频文件过短。长度 {waveform.shape[1]} < 所需长度 {segment_length_samples}")
        else:
            print("✅ 音频文件长度足够。")
        
        # ... 其他处理步骤 ...
        print("✅ 成功获取并处理了第一个样本！")

    except Exception as e:
        # 这一步是关键！打印出具体的错误信息
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"!!! 捕获到错误：在处理第一个文件时失败了 !!!")
        print(f"!!! 具体的错误信息是: {e}")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

else:
    print("❌ 警告：训练集为空，没有找到任何文件。")

print("\n--- 调试结束 ---")
