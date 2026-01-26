import torchaudio, librosa, torch, random, functools
from torch.utils.data import Dataset
import numpy as np
from spec_aug import audio_mask_nosave
import os
from RawBoost import ISD_additive_noise, LnL_convolutive_noise, SSI_additive_noise, normWav


def collate_fn_varlen(batch):
    """
    Collate function for variable-length audio samples.

    Args:
        batch: List of (waveform, label) or (waveform, label, path) tuples.
               Items may be None (failed loading), which will be filtered out.

    Returns:
        padded_waveforms: Tensor of shape (B, T_max), zero-padded to the max length in the batch.
        labels: Tensor of shape (B,).
        lengths: Tensor of shape (B,), original lengths before padding.
        paths: (optional) List of paths if the input tuples include paths.
    """
    # Drop failed samples
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    # Check whether we also return file paths
    if len(batch[0]) == 3:
        waveforms, labels, paths = zip(*batch)
        has_paths = True
    else:
        waveforms, labels = zip(*batch)
        paths = None
        has_paths = False

    # Original lengths (assumes waveform is 1D tensor: (T,))
    lengths = torch.tensor([w.size(0) for w in waveforms], dtype=torch.long)

    # Pad to max length within the batch
    max_len = lengths.max().item()
    padded_waveforms = torch.zeros(len(waveforms), max_len, dtype=waveforms[0].dtype)
    for i, w in enumerate(waveforms):
        padded_waveforms[i, :w.size(0)] = w

    labels = torch.tensor(labels, dtype=torch.long)

    if has_paths:
        return padded_waveforms, labels, lengths, paths
    return padded_waveforms, labels, lengths


def process_Rawboost_feature(feature, sr, args, algo):
    """
    Apply RawBoost-style waveform augmentations.

    algo:
        1: Convolutive noise
        2: Impulsive noise
        3: Colored additive noise
        4: 1+2+3 in series
        5: 1+2 in series
        6: 1+3 in series
        7: 2+3 in series
        8: 1||2 in parallel
        else: no augmentation
    """
    if algo == 1:
        feature = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG,
            args.minBiasLinNonLin, args.maxBiasLinNonLin, sr
        )
    elif algo == 2:
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
    elif algo == 3:
        feature = SSI_additive_noise(
            feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr
        )
    elif algo == 4:
        feature = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG,
            args.minBiasLinNonLin, args.maxBiasLinNonLin, sr
        )
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(
            feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr
        )
    elif algo == 5:
        feature = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG,
            args.minBiasLinNonLin, args.maxBiasLinNonLin, sr
        )
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
    elif algo == 6:
        feature = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG,
            args.minBiasLinNonLin, args.maxBiasLinNonLin, sr
        )
        feature = SSI_additive_noise(
            feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr
        )
    elif algo == 7:
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(
            feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr
        )
    elif algo == 8:
        feature1 = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG,
            args.minBiasLinNonLin, args.maxBiasLinNonLin, sr
        )
        feature2 = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = normWav(feature1 + feature2, 0)  # normalize the combined waveform

    return feature


def count_calls(counter_dict, key):
    """Decorator to count how many times a function is called."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            counter_dict[key] += 1
            return func(*args, **kwargs)
        return wrapper
    return decorator


class Default_Augment:
    def __init__(self, args, probability=None):
        self.sr = 16000
        self.args = args

    def rawboost(self, waveform):
        waveform = waveform.squeeze(0)
        rb = process_Rawboost_feature(feature=waveform, sr=self.sr, args=self.args, algo=5)
        rb = torch.tensor(rb)
        return rb.unsqueeze(0)

    def spec_augment(self, waveform):
        waveform = waveform.squeeze(0)
        wav_np = waveform.numpy()
        aug_np = audio_mask_nosave(audio=wav_np)
        return torch.tensor(aug_np).unsqueeze(0)

    def noise(self, waveform):
        waveform = waveform.squeeze(0)

        # White noise with the same length as the signal
        noise = torch.randn_like(waveform)

        # Random SNR in [13, 20] dB
        snr_db = random.uniform(13, 20)
        snr_linear = 10 ** (snr_db / 10)

        # Power (clamp to avoid division by zero)
        signal_power = waveform.pow(2).mean().clamp_min(1e-12)
        noise_power = noise.pow(2).mean().clamp_min(1e-12)

        desired_noise_power = signal_power / snr_linear
        noise_gain = torch.sqrt(desired_noise_power / noise_power)

        return (waveform + noise * noise_gain).unsqueeze(0)

    def music(self, waveform):
        waveform = waveform.squeeze(0)
        musan_noise_dir = '/path/to/musan/music'  # Root directory of MUSAN music noise (set to your local path)
        noise_list = []

        # Collect all .wav files under the noise directory
        for root, dirs, files in os.walk(musan_noise_dir):
            for file in files:
                if file.endswith(".wav"):
                    noise_list.append(os.path.join(root, file))

        # Randomly pick one noise file
        noise_file = noise_list[random.randint(0, len(noise_list) - 1)]
        noise, _ = librosa.load(noise_file, sr=self.sr)

        # Match noise length to waveform length by repeating and slicing
        waveform_length = len(waveform)
        noise_length = len(noise)
        repeat_times = (waveform_length // noise_length) + 2
        tiled_noise = list(noise) * repeat_times

        # Random crop for noise segment
        start = random.randint(0, len(tiled_noise) - waveform_length)
        noisy_segment = tiled_noise[start:start + waveform_length]

        # Mix with a random SNR in [13, 20] dB
        snr_db = random.uniform(13, 20)
        snr_linear = 10 ** (snr_db / 10)

        # Compute power and scale noise to the target SNR
        noise_power = np.mean(np.square(noisy_segment))
        signal_power = np.mean(np.square(waveform.numpy()))
        desired_noise_power = signal_power / snr_linear
        noise_gain = np.sqrt(desired_noise_power / noise_power)

        new_waveform = waveform + torch.tensor(noisy_segment, dtype=torch.float32) * noise_gain
        return new_waveform.unsqueeze(0)

    def speech(self, waveform):
        waveform = waveform.squeeze(0)
        musan_noise_dir = '/path/to/musan/speech'  # Root directory of MUSAN speech noise (set to your local path)
        noise_list = []

        # Collect all .wav files under the noise directory
        for root, dirs, files in os.walk(musan_noise_dir):
            for file in files:
                if file.endswith(".wav"):
                    noise_list.append(os.path.join(root, file))

        # Randomly pick one noise file
        noise_file = noise_list[random.randint(0, len(noise_list) - 1)]
        noise, _ = librosa.load(noise_file, sr=self.sr)

        # Match noise length to waveform length by repeating and slicing
        waveform_length = len(waveform)
        noise_length = len(noise)
        repeat_times = (waveform_length // noise_length) + 2
        tiled_noise = list(noise) * repeat_times

        # Random crop for noise segment
        start = random.randint(0, len(tiled_noise) - waveform_length)
        noisy_segment = tiled_noise[start:start + waveform_length]

        # Mix with a random SNR in [13, 20] dB
        snr_db = random.uniform(13, 20)
        snr_linear = 10 ** (snr_db / 10)

        # Compute power and scale noise to the target SNR
        noise_power = np.mean(np.square(noisy_segment))
        signal_power = np.mean(np.square(waveform.numpy()))
        desired_noise_power = signal_power / snr_linear
        noise_gain = np.sqrt(desired_noise_power / noise_power)

        new_waveform = waveform + torch.tensor(noisy_segment, dtype=torch.float32) * noise_gain
        return new_waveform.unsqueeze(0)


    def Time_Stretch(self, waveform):
        waveform=waveform.squeeze(0)
        waveform_numpy = waveform.numpy()
        min_rate = 0.8
        max_rate = 1.2
        stretch_factor=random.uniform(min_rate, max_rate)
        wav_stretched_np = librosa.effects.time_stretch(y=waveform_numpy, rate=stretch_factor)
        wav_stretched = torch.from_numpy(wav_stretched_np).unsqueeze(0)
        return wav_stretched

    def none_transform(self, waveform):
        return waveform

    def __call__(self, waveform):
        # With small probability, do not apply augmentation
        if np.random.rand() < 0.1:
            return waveform
        augmentations = [self.rawboost, self.spec_augment, self.noise]
        augmentation = np.random.choice(augmentations)
        return augmentation(waveform)

    def get_call_counts(self):
        return self.call_counts


class Default_dataset(Dataset):
    def __init__(
        self, prctl_path, sample_rate=16000, segment_time=4, transform=None, return_path=None,
        split='train', type='both', variable_length=False, min_len=1.0, max_len=8.0, max_length=None
    ):
        """
        Dataset wrapper for audio + labels parsed from a protocol file.

        Args:
            prctl_path: Path to a protocol file (one sample per line).
            sample_rate: Target sampling rate.
            segment_time: Fixed segment length (seconds) when variable_length=False.
            transform: Optional waveform transform/augmentation.
            return_path: If True, return (waveform, label, path).
            split: One split or a list of splits to keep.
            type: 'both', 'bonafide', or 'spoof'.
            variable_length: If True, keep variable lengths within [min_len, max_len].
            min_len/max_len: Min/max length (seconds) for variable-length mode.
            max_length: If set, randomly subsample up to this many entries.
        """
        self.prctl_path = prctl_path
        self.sample_rate = sample_rate
        self.segment_length = segment_time * sample_rate
        self.transform = transform
        self.return_path = return_path
        self.variable_length = variable_length
        self.min_len = int(min_len * sample_rate)
        self.max_len = int(max_len * sample_rate)

        with open(prctl_path, 'r') as p:
            protocol = p.readlines()

        splits = split if isinstance(split, list) else [split]
        valid_splits = {'train', 'dev', 'eval', 'test', '-'}

        for sp in splits:
            if sp not in valid_splits:
                raise ValueError(f"Invalid split '{sp}'. Must be one of: {', '.join(valid_splits)}")

        # Keep only requested splits (split field assumed at column index 8)
        protocol = [item for item in protocol if item.split(' ')[8].strip() in splits]

        if type == 'bonafide':
            protocol = [item for item in protocol if item.split(' ')[4] == 'bonafide']
        elif type == 'spoof':
            protocol = [item for item in protocol if item.split(' ')[4] == 'spoof']
        elif type != 'both':
            raise ValueError("Invalid type. Must be one of: 'both', 'bonafide', 'spoof'.")

        if max_length is not None:
            protocol = random.sample(protocol, max_length)

        self.audio_files = [item.split(' ')[5] for item in protocol]
        labels_raw = [item.split(' ')[4] for item in protocol]

        # Map labels: spoof -> 0, bonafide -> 1
        clean_audio_files, clean_labels = [], []
        for audio_file, label in zip(self.audio_files, labels_raw):
            if label in ['bonafide', 'spoof']:
                clean_audio_files.append(audio_file)
                clean_labels.append(0 if label == 'spoof' else 1)
            else:
                print(audio_file, 'has invalid label:', label, ', discarded')

        self.audio_files = clean_audio_files
        self.labels = clean_labels

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        if idx >= len(self.audio_files) or idx < 0:
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self.audio_files)}")

        audio_file = self.audio_files[idx]

        try:
            waveform, sr = librosa.load(audio_file, sr=16000, mono=True, dtype=np.float32)
            waveform = torch.tensor(waveform).unsqueeze(0)
        except Exception as e:
            print(f"Error loading {audio_file}: {e}")
            return None

        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)

        if self.transform:
            waveform = self.transform(waveform)

        waveform_length = waveform.size(1)

        if self.variable_length:
            # Enforce length range by repeat/crop
            if waveform_length < self.min_length:
                repeats = self.min_length // waveform_length + 1
                waveform = waveform.repeat(1, repeats)[:, :self.min_length]
            elif waveform_length > self.max_length:
                start = random.randint(0, waveform_length - self.max_length)
                waveform = waveform[:, start:start + self.max_length]
        else:
            # Fixed-length segment by repeat/crop
            if waveform_length < self.segment_length:
                repeats = self.segment_length // waveform_length + 1
                waveform = waveform.repeat(1, repeats)[:, :self.segment_length]
            else:
                start = random.randint(0, waveform_length - self.segment_length)
                waveform = waveform[:, start:start + self.segment_length]

        label = self.labels[idx]

        if self.return_path:
            return waveform.squeeze(0), label, audio_file
        return waveform.squeeze(0), label
