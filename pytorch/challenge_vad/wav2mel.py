import torchaudio, torch
import os, argparse, pickle, pdb
from glob import glob
from tqdm import tqdm
import concurrent.futures, multiprocessing
args = argparse.ArgumentParser()

args.add_argument('--nfft', type=int, default=1024)
args.add_argument('--nmels', type=int, default=80)
args.add_argument('--win', type=int, default=25, help='ms')
args.add_argument('--shift', type=int, default=10, help='ms')
args.add_argument('--sr', type=int, default=16000, help='hz')
args.add_argument('--cpu', type=int, default=multiprocessing.cpu_count() // 2, help='using number of cpu')

config = args.parse_args()


def main(config):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    ABSpath = '/root/datasets'
    # wave load
    path = os.path.join(ABSpath, 'ai_challenge/TEDLIUM-3/TEDLIUM_release-3/data/wav')
    data_path = sorted(glob(path+'/*.wav'))
    save_path = os.path.join(ABSpath, f'ai_challenge/TEDLIUM-3/TEDLIUM_release-3/data/mel//tedrium_nfft{config.nfft}_win{config.win}_hop{config.shift}_nmel{config.nmels}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    win_length = config.win * config.sr // 1000 # ms
    hop_length = config.shift * config.sr // 1000 # ms
    wav2mel = torchaudio.transforms.MelSpectrogram(sample_rate=config.sr, n_fft=config.nfft, win_length=win_length, hop_length=hop_length, n_mels=config.nmels)
    def wavtomel(path):
        wav, sr = torchaudio.load_wav(path)
        
        if sr != config.sr:
            raise ValueError('sampling rate is different from config.sr')
        
        # wav = (channel, time)
        mel = wav2mel(wav)
        with open(os.path.join(save_path, path.split('/')[-1].split('.')[0]) + '.pickle', 'wb') as f:
            pickle.dump(mel.numpy())

    with concurrent.futures.ProcessPoolExecutor(max_workers=config.cpu) as executor:
        executor.map(wavtomel, data_path)
    print('all process is done')

        

if __name__ == "__main__":
    main(config)
