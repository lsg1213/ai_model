import os, pickle
import glob
import torch
import random, pdb
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from utils.utils import read_wav_np, cut_wav, get_length, process_blizzard
from utils.audio import MelGen
from utils.tierutil import TierUtil

def create_dataloader(hp, args, train):
    # if args.tts:
    #     dataset = AudioTextDataset(hp, args, train)
    #     return DataLoader(
    #         dataset=dataset,
    #         batch_size=args.batch_size,
    #         shuffle=train,
    #         num_workers=hp.train.num_workers,
    #         pin_memory=True,
    #         drop_last=True,
    #         collate_fn=TextCollate()
    #     )
    # else:
    dataset = AudioOnlyDataset(hp, args, train)
    return DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=hp.train.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=AudioCollate()
    )

def dataSplit(data, args, hp):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # data shape list(25, np(987136, 12)), accel, 주의: 샘플별로 안 섞이게 하기
    # 이걸 자르기, (index, window, channel)
    # data_length = int(hp.audio.sr * hp.audio.win_length / 1000000)
    data_length = int(hp.audio.sr * 1.0)
    splited_data = torch.cat([torch.cat([torch.from_numpy(_data[idx:idx+data_length][np.newaxis, ...]) for idx in range(len(_data) // data_length)]) for _data in data])
    
    return splited_data.cpu()

class AudioOnlyDataset(Dataset):
    def __init__(self, hp, args, train):
        self.hp = hp
        self.args = args
        self.train = train
        self.accel_data = dataSplit(pickle.load(open(hp.accel_data.path, 'rb'))[:2], args, hp)
        self.sound_data = dataSplit(pickle.load(open(hp.sound_data.path, 'rb'))[:2], args, hp)
        self.melgen = MelGen(hp)
        self.tierutil = TierUtil(hp)
        # this will search all files within hp.data.path

        # self.wavlen = int(hp.audio.sr * hp.audio.duration)
        self.tier = self.args.tier
        self.melgen = MelGen(hp)
        self.tierutil = TierUtil(hp)

    def __len__(self):
        return len(self.accel_data)

    def __getitem__(self, idx):
        wav = self.accel_data[idx].cpu().numpy()
        mel = self.melgen.get_normalized_mel(wav)
        source, target = self.tierutil.cut_divide_tiers(mel, self.tier)
        return source, target

# class AudioTextDataset(Dataset):
#     def __init__(self, hp, args, train):
#         self.hp = hp
#         self.args = args
#         self.train = train
#         self.data = hp.data.path
#         self.melgen = MelGen(hp)
#         self.tierutil = TierUtil(hp)

#         # this will search all files within hp.data.path
#         self.root_dir = hp.data.path
#         self.dataset = []
#         if hp.data.name == 'KSS':
#             with open(os.path.join(self.root_dir, 'transcript.v.1.3.txt'), 'r') as f:
#                 lines = f.read().splitlines()
#                 for line in tqdm(lines):
#                     wav_name, _, _, text, length, _ = line.split('|')

#                     wav_path = os.path.join(self.root_dir, 'kss', wav_name)
#                     duraton = float(length)
#                     if duraton < hp.audio.duration:
#                         self.dataset.append((wav_path, text))

#                 # if len(self.dataset) > 100: break
#         elif hp.data.name == 'Blizzard':
#             with open(os.path.join(self.root_dir, 'prompts.gui'), 'r') as f:
#                 lines = f.read().splitlines()
#                 filenames = lines[::3]
#                 sentences = lines[1::3]
#                 for filename, sentence in tqdm(zip(filenames, sentences), total=len(filenames)):
#                     wav_path = os.path.join(self.root_dir, 'wavn', filename + '.wav')
#                     length = get_length(wav_path, hp.audio.sr)
#                     if length < hp.audio.duration:
#                         self.dataset.append((wav_path, sentence))
#         else:
#             raise NotImplementedError

#         random.seed(123)
#         random.shuffle(self.dataset)
#         if train:
#             self.dataset = self.dataset[:int(0.95 * len(self.dataset))]
#         else:
#             self.dataset = self.dataset[int(0.95 * len(self.dataset)):]

#         self.wavlen = int(hp.audio.sr * hp.audio.duration)
#         self.tier = self.args.tier

#         self.melgen = MelGen(hp)
#         self.tierutil = TierUtil(hp)

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         text = self.dataset[idx][1]
#         if self.hp.data.name == 'KSS':
#             seq = text_to_sequence(text)
#         elif self.hp.data.name == 'Blizzard':
#             seq = process_blizzard(text)

        # wav = read_wav_np(self.dataset[idx][0], sample_rate=self.hp.audio.sr)
        # # wav = cut_wav(self.wavlen, wav)
        # mel = self.melgen.get_normalized_mel(wav)
        # source, target = self.tierutil.cut_divide_tiers(mel, self.tier)

        # return seq, source, target

class TextCollate():
    def __init__(self):
        return

    def __call__(self, batch):
        seq = [torch.from_numpy(x[0]).long() for x in batch]
        text_lengths = torch.LongTensor([x.shape[0] for x in seq])
        # Right zero-pad all one-hot text sequences to max input length
        seq_padded = torch.nn.utils.rnn.pad_sequence(seq, batch_first=True)

        audio_lengths = torch.LongTensor([x[1].shape[1] for x in batch])
        source_padded = torch.nn.utils.rnn.pad_sequence(
            [torch.from_numpy(x[1].T) for x in batch],
            batch_first=True
        ).transpose(1, 2)
        target_padded = torch.nn.utils.rnn.pad_sequence(
            [torch.from_numpy(x[2].T) for x in batch],
            batch_first=True
        ).transpose(1, 2)

        return seq_padded, text_lengths, source_padded, target_padded, audio_lengths

class AudioCollate():
    def __init__(self):
        return

    def __call__(self, batch):
        audio_lengths = torch.LongTensor([x[0].shape[1] for x in batch])
        source_padded = torch.nn.utils.rnn.pad_sequence(
            [torch.from_numpy(x[0].T) for x in batch],
            batch_first=True
        ).transpose(1, 2)
        target_padded = torch.nn.utils.rnn.pad_sequence(
            [torch.from_numpy(x[1].T) for x in batch],
            batch_first=True
        ).transpose(1, 2)

        return source_padded, target_padded, audio_lengths
