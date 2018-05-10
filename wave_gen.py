import librosa
import config
from network.wavenet import WaveNet
import datasets
import numpy as np
import torch
from torch.autograd import Variable



def _variable(data):
    tensor = torch.from_numpy(data).float()

    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)



def get_seed_from_audio(filepath, rfield_size, sample_size):
    audio = datasets.get_wave(filepath)#, sample_rate)
    audio_length = len(audio)

    audio = np.pad([audio], [[0, 0], [rfield_size, 0], [0, 0]], 'constant')
    if sample_size:
        seed = audio[:, :sample_size, :]
    else:
        seed = audio[:, :rfield_size*2, :]

    return _variable(seed), audio_length


def save_to_audio_file(data):
    sample_rate = 16000
    data = data[0].cpu().data.numpy()
    data = datasets.one_hot_decode(data, axis=1)
    audio = datasets.mu_law_decode(data, in_out_size)

    librosa.output.write_wav('piano.wav', audio, sample_rate)
    print('Saved wav file at {}'.format('piano.wav'))

    return librosa.get_duration(y=audio, sr=sample_rate)


