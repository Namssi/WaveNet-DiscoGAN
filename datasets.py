from __future__ import print_function


import torch
import torch.utils.data as data
import os
import numpy as np
import librosa


def mu_law_encode(wave, quantization_size=256):
    mu = float(quantization_size-1)
    quantize_space = np.linspace(-1, 1, quantization_size)

    quantized = np.sign(wave) * np.log(1 + mu * np.abs(wave)) / np.log(mu + 1)
    quantized = np.digitize(quantized, quantize_space) - 1
    return quantized

def mu_law_decode(encoded, quantization_size=256):
    mu = float(quantization_size - 1)
    quantization_size = float(quantization_size)
    expanded = (encoded / quantization_size) * 2. - 1
    wave = np.sign(expanded) * (np.exp(np.abs(expanded) * np.log(mu + 1)) - 1) / mu

    return wave

def one_hot_encode(data, channels=256):
    #print(data, data.size, data.shape)
    one_hot = np.zeros((data.size, channels), dtype=float)
    one_hot[np.arange(data.size), data.ravel()] = 1

    return one_hot

def one_hot_decode(data, axis=1):
    decoded = np.argmax(data, axis=axis)

    return decoded


def get_wave(wave_file):
    #wave48 = librosa.load(wave_file, mono=True, sr=None)
    #wave16 = wave48[::3]
    #preprocessed_wave = librosa.feature.mfcc(wave16, sr=16000)

    wave,_ = librosa.load(wave_file, mono=True, sr=16000)
    #print('librosa.load: ', wave, wave.shape)
    wave = wave.reshape(-1,1)#transpose to vector shapei
    #print('reshape: ',wave, len(wave), wave.shape)
    #wave16,_ = librosa.effects.trim(wave16)
    #print('librosa.effects: ',wave16, len(wave16))
    preprocessed_wave = mu_law_encode(wave)
    preprocessed_wave = one_hot_encode(preprocessed_wave)
    #print('preprocessed: ', preprocessed_wave, preprocessed_wave.shape)

    return preprocessed_wave


class WaveDataset(data.Dataset):
    def __init__(self, data_dir_1, data_dir_2, sample_rate, wave_size):
	
	self.data_dir_s = data_dir_1
        self.data_dir_t = data_dir_2
	self.file_names_s = [x for x in sorted(os.listdir(data_dir_1))]
        self.file_names_t = [x for x in sorted(os.listdir(data_dir_2))]



    def __len__(self):
	return len(self.file_names_s)



    def __getitem__(self, index):
	#key = self.file_names[index][:-4]

	file_path_s = os.path.join(self.data_dir_s, self.file_names_s[index])
        file_path_t = os.path.join(self.data_dir_t, self.file_names_t[index])
        #raw_audio = load_audio(filepath, self.sample_rate, self.trim)
	wave_s = get_wave(file_path_s)
        wave_t = get_wave(file_path_t)

        #cut & merge
        shape = np.minimum(wave_s.shape[0], wave_t.shape[0])
        wave_s = wave_s[:shape,:]
        wave_t = wave_t[:shape,:]
        #print('wave_s.shape: ', wave_s.shape)
        #print('wave_t.shape: ', wave_t.shape)
        wave = np.concatenate((wave_s, wave_t), axis=0)
        #print('wave.shape: ', wave.shape)

	return wave	






class DataLoader(data.DataLoader):
    def __init__(self, data_dir_1, data_dir_2, receptive_fields,
                 sample_size=0, sample_rate=16000, in_channels=256,
                 batch_size=1, shuffle=True):
        """
        DataLoader for WaveNet
        :param data_dir:
        :param receptive_fields: integer. size(length) of receptive fields
        :param sample_size: integer. number of timesteps to train at once.
                            sample size has to be bigger than receptive fields.
                            |-- receptive field --|---------------------|
                            |------- samples -------------------|
                            |---------------------|-- outputs --|
        :param sample_rate: sound sampling rates
        :param in_channels: number of input channels
        :param batch_size:
        :param shuffle:
        """
        dataset = WaveDataset(data_dir_1, data_dir_2, sample_rate, in_channels)

        super(DataLoader, self).__init__(dataset, batch_size, shuffle)

        if sample_size <= receptive_fields:
            raise Exception("sample_size has to be bigger than receptive_fields")

        self.sample_size = sample_size
        self.receptive_fields = receptive_fields

        self.collate_fn = self._collate_fn

    def calc_sample_size(self, audio):
        return self.sample_size if len(audio[0]) >= self.sample_size\
                                else len(audio[0])

    @staticmethod
    def _variable(data):
        tensor = torch.from_numpy(data).float()

        if torch.cuda.is_available():
            return torch.autograd.Variable(tensor.cuda())
        else:
            return torch.autograd.Variable(tensor)

    def _collate_fn(self, audio):#type(audio): list
        #print('_callate_fn', len(audio))
        shape = len(audio[0])/2
        #print(audio, type(audio), len(audio), len(audio[0]))
        audio1 = [np.array(audio[0][:shape])]
        audio2 = [np.array(audio[0][shape:])]
        #print('audio1: ', audio1, len(audio1), len(audio1[0]))
        #print('audio2: ', audio2, len(audio2), len(audio2[0]))

        audio1 = np.pad(audio1, [[0, 0], [self.receptive_fields, 0], [0, 0]], 'constant')
        audio2 = np.pad(audio2, [[0, 0], [self.receptive_fields, 0], [0, 0]], 'constant')
        #print('_callate_fn:pad', audio.shape)

        if self.sample_size:
            sample_size_1 = self.calc_sample_size(audio1)
            sample_size_2 = self.calc_sample_size(audio2)

            while (sample_size_1 > self.receptive_fields) and (sample_size_2 > self.receptive_fields):
                inputs1 = audio1[:, :sample_size_1, :]
                targets1 = audio1[:, self.receptive_fields:sample_size_1, :]
                inputs2 = audio2[:, :sample_size_2, :]
                targets2 = audio2[:, self.receptive_fields:sample_size_2, :]

                yield self._variable(inputs1), self._variable(targets1),\
                      self._variable(inputs2), self._variable(targets2)
                #yield self._variable(inputs1), self._variable(one_hot_decode(targets1, 2)),\
                      #self._variable(inputs2), self._variable(one_hot_decode(targets2, 2))

                audio1 = audio1[:, sample_size_1-self.receptive_fields:, :]
                sample_size_1 = self.calc_sample_size(audio1)
                audio2 = audio2[:, sample_size_2-self.receptive_fields:, :]
                sample_size_2 = self.calc_sample_size(audio2)


        else:
            targets1 = audio1[:, self.receptive_fields:, :]
            targets2 = audio2[:, self.receptive_fields:, :]
            yield self._variable(audio1), self._variable(targets1),\
                  self._variable(audio2), self._variable(targets2)
            #yield self._variable(audio1), self._variable(one_hot_decode(targets1, 2)),\
                  #self._variable(audio2), self._variable(one_hot_decode(targets2, 2))


