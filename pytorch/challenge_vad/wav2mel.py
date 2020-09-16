import torchaudio, torch
import numpy as np
import os, argparse, joblib, pdb
from glob import glob
from tqdm import tqdm
import concurrent.futures, multiprocessing
args = argparse.ArgumentParser()

args.add_argument('--nfft', type=int, default=1024)
args.add_argument('--nmels', type=int, default=80)
args.add_argument('--win', type=int, default=25, help='ms')
args.add_argument('--hop', type=int, default=10, help='ms')
args.add_argument('--sr', type=int, default=16000, help='hz')
args.add_argument('--cpu', type=int, default=multiprocessing.cpu_count() // 2, help='using number of cpu')

config = args.parse_args()


def main(config):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    ABSpath = '/root/datasets/ai_challenge'
    data_path = ABSpath + '/TEDLIUM-3/TEDLIUM_release-3/data'
    label_path = data_path + '/label'
    # wave load
    path = os.path.join(ABSpath, data_path + '/wav')
    wav_path = sorted(glob(path+'/*.wav'))
    label_path = sorted(glob(label_path + '/*'))
    save_path = os.path.join(ABSpath, f'TEDLIUM-3/TEDLIUM_release-3/data/mel/tedrium_nfft{config.nfft}_win{config.win}_hop{config.hop}_nmel{config.nmels}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    win_length = config.win * config.sr // 1000 # ms
    hop_length = config.hop * config.sr // 1000 # ms
    
    
    

    def datatomel(path):
        def wavtomel(path):
            wav, sr = torchaudio.load_wav(path)
            
            if sr != config.sr:
                raise ValueError('sampling rate is different from config.sr')
            
            # wav = (channel, time)
            
            mel = torchaudio.transforms.MelSpectrogram(sample_rate=config.sr, n_fft=config.nfft, win_length=win_length, hop_length=hop_length, n_mels=config.nmels)(wav)
            return mel

        def labeltowindow(path):
            label = np.load(path)
            winlabel = np.concatenate([label[hop_length * t:hop_length * t + win_length][np.newaxis, ...] for t in range(label.shape[0] // hop_length - 1)])
            return winlabel

        wa, la = path
        if wa.split('/')[-1].split('.')[0] != la.split('/')[-1].split('.')[0]:
            raise ValueError('data is not matched')
        mel = wavtomel(wa).squeeze(0)
        lab = torch.from_numpy(labeltowindow(la)).transpose(0,1)
        mel = mel[:,:lab.size(-1)]
        return (mel, lab)
    with concurrent.futures.ProcessPoolExecutor(max_workers=config.cpu) as executor:
        mel, lab = executor.map(datatomel, zip(wav_path, label_path))
        with open(os.path.join(save_path, path.split('/')[-1].split('.')[0]) + '.joblib', 'wb') as f:
            joblib.dump((mel.numpy(), lab.numpy()), f)

    
        



    ############### label load #################
    # with concurrent.futures.ProcessPoolExecutor(max_workers=config.cpu) as executor:
    #     executor.map(wavtomel, data_path)
    


if __name__ == "__main__":
    main(config)
    # kk()