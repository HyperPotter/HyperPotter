import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf


def time_masking(spectrogram, time_masking_para=100, time_mask_num=2):
    
    tau = spectrogram.shape[1]

    if tau < time_masking_para:
        return spectrogram
    else:
        for i in range(time_mask_num):
            t = np.random.randint(0, time_masking_para)
            t0 = np.random.randint(0, tau - t)
            spectrogram[:, t0:t0 + t] = 0
        return spectrogram

def frequency_masking(spectrogram, frequency_masking_para=100, frequency_mask_num=2):

    tau = spectrogram.shape[0]

    for i in range(frequency_mask_num):
        t = np.random.randint(0, frequency_masking_para)
        t0 = np.random.randint(0, tau - t)
        spectrogram[t0:t0 + t,: ] = 0
    return spectrogram


def audio_show(wav_path):
    audio, sr = librosa.load(wav_path, sr=None)
    filename=wav_path.split('/')[-1].split('.')[0]
    print(filename)
    n_fft = 1024
    hop_length = 128
    stft_result = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(stft_result)

    plt.figure(figsize=(10, 6))
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max),
                            sr=sr, hop_length=hop_length, y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'{filename}')
    plt.savefig('mask_resfft.png')

def compare_wav(one_path ,two_path):
    audio_one, sr = librosa.load(one_path)
    audio_two, sr = librosa.load(two_path)
    count=0
    print(audio_one)
    print(audio_two)
    print(len(audio_one))
    print(len(audio_two))
    for i in range(len(audio_two)):
        if audio_one[i]!=audio_two[i]:
            count+=1
    print(count)
    print(len(audio_one))

def audio_mask(audio,save_path,sr,time_masking_para=20,fre_masking_para=10,time_mask_num=5,fre_mask_num=5):
    n_fft = 1024
    hop_length = 128
    stft_result = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)

    spectrogram = np.abs(stft_result)

    masked_spectrogram=time_masking(spectrogram,time_masking_para,time_mask_num)
    masked_spectrogram=frequency_masking(masked_spectrogram,fre_masking_para,fre_mask_num)

    masked_stft_result = masked_spectrogram * np.exp(1j * np.angle(stft_result))
    reconstructed_audio = librosa.istft(masked_stft_result, hop_length=hop_length)

    sf.write(save_path, reconstructed_audio,sr,subtype='FLOAT')
    return reconstructed_audio

def audio_mask_nosave(audio,time_masking_para=20,fre_masking_para=10,time_mask_num=5,fre_mask_num=5):
    n_fft = 1024
    hop_length = 128
    stft_result = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)

    spectrogram = np.abs(stft_result)

    masked_spectrogram=time_masking(spectrogram,time_masking_para,time_mask_num)
    masked_spectrogram=frequency_masking(masked_spectrogram,fre_masking_para,fre_mask_num)

    masked_stft_result = masked_spectrogram * np.exp(1j * np.angle(stft_result))
    reconstructed_audio = librosa.istft(masked_stft_result, hop_length=hop_length)

    return reconstructed_audio