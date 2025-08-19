# dataset.py
import os
import random
import glob
import torch
import torchaudio
from torch.utils.data import Dataset
# from perturb import Perturbation # 保留，为后续做准备

class LibriSpeechSpeakerDataset(Dataset):
    def __init__(self, config, train=True, apply_perturbation=False):
        self.config = config
        self.data_path = config['data']['dataset_path']
        self.train = train
        self.apply_perturbation = apply_perturbation

        # 1. 获取文件列表并构建 speaker-to-id 映射
        self.file_list, self.speaker_to_id = self._get_file_list_and_speaker_map()
        self.id_to_speaker = {i: s for s, i in self.speaker_to_id.items()}
        print(f"Found {len(self.speaker_to_id)} unique speakers.")

        # 2. 划分训练集和验证集
        random.seed(42) # 固定随机种子以保证每次划分一致
        random.shuffle(self.file_list)
        split_idx = int(len(self.file_list) * (1 - config['data']['val_split_ratio']))
        
        if self.train:
            self.file_list = self.file_list[:split_idx]
            print(f"Found {len(self.file_list)} files for training.")
        else:
            self.file_list = self.file_list[split_idx:]
            print(f"Found {len(self.file_list)} files for validation.")

        # 3. 音频处理参数
        self.sample_rate = config['audio']['sample_rate']
        self.segment_length_ms = config['audio']['segment_length_ms']
        self.segment_length_samples = int(self.sample_rate * self.segment_length_ms / 1000)

        # 4. 在初始化时创建 Mel 频谱图转换器
        self.mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=config['audio']['n_fft'],
            win_length=config['audio']['win_length'],
            hop_length=config['audio']['hop_length'],
            n_mels=config['audio']['n_mels']
        )

        # 5. 初始化扰动模块 (如果需要)
        # if self.train and self.apply_perturbation:
        #     self.perturbation = Perturbation(config)

    def _get_file_list_and_speaker_map(self):
        file_list = []
        speaker_to_id = {}
        
        # 递归扫描所有 .flac 文件
        print(f"Scanning for .flac files in {self.data_path}...")
        wav_files = glob.glob(os.path.join(self.data_path, "**", "*.flac"), recursive=True)
        print(f"Found {len(wav_files)} total .flac files.")
        
        for file_path in wav_files:
            # 从路径中提取 speaker_id (e.g., .../train-clean-100/19/...)
            try:
                speaker_id_str = file_path.split(os.sep)[-3]
            except IndexError:
                # print(f"Warning: Could not extract speaker ID from path: {file_path}")
                continue

            if speaker_id_str not in speaker_to_id:
                speaker_to_id[speaker_id_str] = len(speaker_to_id)
            
            speaker_id = speaker_to_id[speaker_id_str]
            file_list.append((file_path, speaker_id))
            
        return file_list, speaker_to_id

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path, speaker_id = self.file_list[idx]

        try:
            waveform, sample_rate = torchaudio.load(file_path)
            
            # 1. 重采样到目标采样率
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
                waveform = resampler(waveform)

            # 2. 如果音频太短，则返回 None，由 collate_fn 过滤
            if waveform.shape[1] < self.segment_length_samples:
                return None

            # 3. 随机截取一个固定长度的片段
            start_idx = random.randint(0, waveform.shape[1] - self.segment_length_samples)
            segment = waveform[:, start_idx : start_idx + self.segment_length_samples]

            # 4. (可选) 应用扰动
            if self.train and self.apply_perturbation:
                segment = self.perturbation(segment)

            # 5. 提取 Mel 频谱图并转换为对数刻度
            mel_spec = self.mel_spectrogram_transform(segment)
            mel_spec = torch.log(mel_spec + 1e-9)

            return mel_spec, speaker_id

        except Exception as e:
            # print(f"Warning: Error loading or processing file {file_path}: {e}")
            return None