import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset
import glob
from perturb import apply_perturbation

class LibriSpeechSpeakerDataset(Dataset):
    def __init__(self, config, train=True):
        self.config = config
        self.data_path = config['data']['dataset_path']
        self.train = train
        
        self.speaker_dirs = sorted(os.listdir(self.data_path))
        self.speaker_map = {speaker_id: i for i, speaker_id in enumerate(self.speaker_dirs)}
        
        self.file_list = []
        for speaker_id_str in self.speaker_dirs:
            speaker_label = self.speaker_map[speaker_id_str]
            # 扫描所有flac文件
            wav_files = glob.glob(os.path.join(self.data_path, speaker_id_str, "*/*.flac"))
            for wav_file in wav_files:
                self.file_list.append((wav_file, speaker_label))

        # 划分训练集和验证集
        random.Random(42).shuffle(self.file_list) # 固定随机种子保证每次划分一致
        split_idx = int(len(self.file_list) * (1 - config['data']['val_split_ratio']))
        if self.train:
            self.file_list = self.file_list[:split_idx]
            print(f"Found {len(self.file_list)} files for training.")
        else:
            self.file_list = self.file_list[split_idx:]
            print(f"Found {len(self.file_list)} files for validation.")

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config['data']['sample_rate'],
            n_fft=config['data']['n_fft'],
            win_length=config['data']['win_length'],
            hop_length=config['data']['hop_length'],
            n_mels=config['data']['num_mels']
        )
        self.segment_length = config['data']['sample_rate'] * config['data']['train_segment_sec']

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        wav_file, speaker_id = self.file_list[index]
        
        try:
            waveform, sample_rate = torchaudio.load(wav_file)
        except Exception as e:
            print(f"Error loading file {wav_file}: {e}")
            return self.__getitem__(random.randint(0, len(self.file_list) - 1))

        if sample_rate != self.config['data']['sample_rate']:
            waveform = torchaudio.transforms.functional.resample(waveform, orig_freq=sample_rate, new_freq=self.config['data']['sample_rate'])

        if waveform.shape[1] < self.segment_length:
            return self.__getitem__(random.randint(0, len(self.file_list) - 1))
        
        start = random.randint(0, waveform.shape[1] - self.segment_length)
        waveform = waveform[:, start:start + self.segment_length]

        # 如果是训练集，并且配置中启用了扰动，则应用扰动
        if self.train and self.config['perturbation']['enabled']:
            waveform = apply_perturbation(waveform, self.config['perturbation']['strength'])
            
        mel_spec = self.mel_spectrogram(waveform)
        mel_spec = mel_spec.squeeze(0).transpose(0, 1)

        return mel_spec, speaker_id