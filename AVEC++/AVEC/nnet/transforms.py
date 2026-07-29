# Copyright 2021, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch
import torch
import torch.nn as nn
import torchaudio

# Other
import numpy as np
import collections
import skimage.transform
import sys
from pathlib import Path

# Face Detectors
try:
    from ibug.face_detection import RetinaFacePredictor
except Exception as e:
    print(e)
try:
    from ibug.face_alignment import FANPredictor
except Exception as e:
    print(e)

###############################################################################
# Transforms
###############################################################################

class NormalizeVideo(nn.Module):

    def __init__(self, mean, std):
        super().__init__()

        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).reshape(len(mean), 1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).reshape(len(std), 1, 1, 1), persistent=False)

    def forward(self, x):

        x = (x - self.mean) / self.std

        return x

class DenormalizeVideo(nn.Module):

    def __init__(self, mean, std):
        super().__init__()

        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).reshape(len(mean), 1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).reshape(len(std), 1, 1, 1), persistent=False)

    def forward(self, x):

        x = x * self.std + self.mean

        return x

def video_to_images(videos):

    # (B, C, T, H, W) -> (B*T, C, H, W)
    return videos.transpose(1, 2).flatten(start_dim=0, end_dim=1)

def images_to_videos(images, video_frames):

    #(B*T, C, H, W) -> (B, C, T, H, W)
    assert images.size(0) % video_frames == 0
    return images.reshape(images.size(0) // video_frames, video_frames, images.size(1), images.size(2), images.size(3)).transpose(1, 2)

class VideoToImages(nn.Module):

    """ Transform Batch of Videos to Batch of Images """

    def __init__(self):
        super().__init__()

    def forward(self, videos):
        return video_to_images(videos)

class ImagesToVideos(nn.Module):

    """ Transform Batch of Images to Batch of Videos """

    def __init__(self, video_frames=None):
        super().__init__()
        self.video_frames = video_frames

    def forward(self, images, video_frames=None):

        # default video_frames
        if video_frames == None:
            video_frames = self.video_frames

        # assert not None
        assert video_frames != None

        return images_to_videos(images, video_frames)

class TimeMaskSecond(nn.Module):

    def __init__(self, T_second, num_mask_second, fps, mean_frame=False):
        super().__init__()

        self.T = int(T_second * fps)
        self.num_mask_second = num_mask_second
        self.mean_frame = mean_frame
        self.fps = fps

    def forward(self, x):

        mT = int(x.shape[-1] / self.fps * self.num_mask_second)

        # Mask
        for _ in range(mT):
            x = torchaudio.functional.mask_along_axis(x, mask_param=self.T, mask_value=x.mean() if self.mean_frame else 0.0, axis=2)

        return x

class BabbleNoise(nn.Module):

    def __init__(self, noise_file_path="datasets/NoiseX/babble/babble.flac", SNR_db=[-5, 0, 5, 10, 15, 20, None], to_sample_rate=16000):
        super().__init__()

        self.noise, self.from_sample_rate = torchaudio.load(noise_file_path)
        self.noise = self.noise[:1]
        if to_sample_rate != None:
            self.noise = torchaudio.functional.resample(self.noise, orig_freq=self.from_sample_rate, new_freq=to_sample_rate)
        self.SNR_db = SNR_db

    def forward(self, x):

        # Select SNR
        SNR_db = self.SNR_db[torch.randint(0, len(self.SNR_db), ())]

        # Transform
        if SNR_db != None:

            # db to power
            SNR = 10 ** (SNR_db / 10)

            # power to ampl
            SNR = SNR ** 0.5

            # Selet Noise
            pos = torch.randint(0, self.noise.shape[-1] - x.shape[-1] + 1, ())
            noise = self.noise[:, pos:pos + x.shape[-1]]

            # Power
            x_power = (x ** 2).sum() / x.shape[-1]
            noise_power = (noise ** 2).sum() / noise.shape[-1]

            # Scale Noise
            noise = 1 / SNR * noise * torch.sqrt(x_power / noise_power)

            # Add Noise
            x = x + noise

        return x

class HardArtificialNoise(nn.Module):

    """ Reproduces team-1's "hard artificial noise" recipe used to build the LRS-VoxMM-style
    degraded benchmarks that AVEC-only and the whisper baselines were evaluated against
    (reverb -> DEMAND background noise -> bandwidth-limiting filter, chained in that order).

    That original recipe (see noise.py) is an *offline* script: it seeds noise deterministically
    per utterance_id and writes one fixed noisy .wav per sample to disk -- exactly what you want
    for a static, reproducible eval benchmark, wrong for training augmentation (no per-epoch
    variety, and writing a second copy of LRS2/3 to the CLAIX filesystem every epoch is a
    non-starter).

    This version runs the same three stages directly on the waveform tensor LRS.__getitem__
    already has in memory -- no file reads/writes beyond loading the DEMAND corpus once -- and
    draws a fresh, unseeded np.random.default_rng() on every call instead of hashing utterance_id.
    Two consequences of that:
      - it's fork-safe: unlike reusing numpy's *global* RNG (which gets duplicated verbatim across
        DataLoader worker processes and would replay the same "random" sequence in every worker),
        a fresh default_rng() pulls new OS entropy each call, so workers never collide
      - "new noise every epoch" needs no extra code: the DataLoader already calls __getitem__ once
        per sample per epoch, so passing this as LRS(..., audio_transform=HardArtificialNoise(...))
        is the entire integration -- nothing needs to track the epoch number

    Pass this as the *training* audio_transform only -- leave evaluation_dataset's audio_transform
    unset so validation still tracks clean LRS2/3. Keep using noise.py's static offline recipe
    separately when you need a fixed, reproducible degraded set to compare checkpoints against
    LRS-VoxMM-style conditions.

    Perf note: the reverb stage runs a full pyroomacoustics image-source room simulation per
    sample, which is noticeably heavier than the other two stages. If data loading starts
    bottlenecking GPU throughput at LRS2/3 scale (watch step time in the training logs), first
    lower reverb_prob before touching num_workers -- and if it's still too slow, switch to
    convolving against a precomputed bank of RIRs instead of simulating rooms live.

    Defaults below match LRS-VoxMM's published "3-dist hard" synthetic distortion split exactly
    (additive noise + reverb + bandwidth limitation applied jointly, i.e. all three probs=1.0):
      - reverb: room length/width sampled in [5, 15] m, height in [2, 6] m, RT60 in [0.4, 1.0] s
      - additive noise (DEMAND): SNR in [-5, 0] dB for "hard" ([5, 15] dB is their "easy" split --
        pass demand_snr_db_range=(5.0, 15.0) for that instead)
      - bandwidth: butter/bessel/cheby1 low-pass, cutoff in {2, 4, 8} kHz (identical across
        easy/hard -- only the SNR range differs between the two)
    filter_order isn't specified in their writeup; left at a reasonable default. Their "noise
    hard"/"noise easy" splits are additive-noise-only (no reverb/bandwidth) -- set
    reverb_prob=0 and bandwidth_prob=0 to match those instead of 3-dist.

    All three probs default to 1.0 here to replicate "jointly applied to every sample," which is
    what the published eval splits do -- but that also means zero clean exposure during training
    if you use this as-is for the whole run.

    For a mix of clean and hard-distorted samples, prefer joint_prob over lowering
    reverb_prob/demand_prob/bandwidth_prob independently: with three *independent* per-stage
    probabilities, e.g. all three at 0.5, the chance of a sample getting all three together (the
    actual "3-dist hard" condition) is only 0.5^3 = 12.5% -- the other ~87.5% is a mix of partial
    combinations (reverb alone, noise+bandwidth without reverb, ...) that don't correspond to any
    of LRS-VoxMM's published splits, i.e. you'd be inventing new distortion combinations rather
    than replicating theirs. joint_prob instead makes one coin flip per sample: with that
    probability all three stages fire together exactly as configured (real "3-dist hard"
    replication), otherwise the sample is left clean -- set the three *_prob args to 1.0 (the
    default here) and use joint_prob for the clean/hard mixing instead.
    """

    def __init__(
        self,
        demand_noise_path,
        sample_rate=16000,
        joint_prob=None,
        reverb_prob=1.0,
        reverb_length_range=(5.0, 15.0),
        reverb_width_range=(5.0, 15.0),
        reverb_height_range=(2.0, 6.0),
        reverb_rt60_range=(0.4, 1.0),
        demand_prob=1.0,
        demand_snr_db_range=(-5.0, 0.0),
        bandwidth_prob=1.0,
        bandwidth_filters=("butter", "bessel", "cheby1"),
        bandwidth_cutoffs_hz=(2000, 4000, 8000),
        bandwidth_filter_order=4,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.joint_prob = joint_prob
        self.reverb_prob = reverb_prob
        self.reverb_length_range = reverb_length_range
        self.reverb_width_range = reverb_width_range
        self.reverb_height_range = reverb_height_range
        self.reverb_rt60_range = reverb_rt60_range
        self.demand_prob = demand_prob
        self.demand_snr_db_range = demand_snr_db_range
        self.bandwidth_prob = bandwidth_prob
        self.bandwidth_filters = list(bandwidth_filters)
        self.bandwidth_cutoffs_hz = list(bandwidth_cutoffs_hz)
        self.bandwidth_filter_order = bandwidth_filter_order

        # Preload every DEMAND track once (per worker process, at construction time) so
        # training doesn't hit the shared CLAIX filesystem again on every __getitem__ call.
        # DEMAND is ~1.3GB total across all environments -- comfortably fits in memory.
        self._tracks = []
        if demand_prob > 0:
            root = Path(demand_noise_path)
            if not root.is_dir():
                raise ValueError(f"DEMAND noise path does not exist or is not a directory: {root}")
            for wav_path in sorted(root.glob("*/*.wav")):
                noise_audio, noise_rate = torchaudio.load(str(wav_path))
                if noise_rate != sample_rate:
                    noise_audio = torchaudio.functional.resample(noise_audio, orig_freq=noise_rate, new_freq=sample_rate)
                self._tracks.append(noise_audio[0].numpy().astype(np.float32))
            if not self._tracks:
                raise ValueError(f"DEMAND noise path contains no WAV files under environment folders: {root}")

    def forward(self, x):

        # x: (1, T) waveform tensor, same convention as BabbleNoise
        rng = np.random.default_rng()
        audio = x[0].numpy().astype(np.float32, copy=True)

        if self.joint_prob is not None:
            # Single per-sample gate: all three stages fire together or none do, so the
            # distorted fraction of samples exactly matches a published split (e.g. 3-dist
            # hard) instead of inventing partial combinations -- see class docstring.
            apply_all = rng.random() < self.joint_prob
            do_reverb = apply_all
            do_demand = apply_all and bool(self._tracks)
            do_bandwidth = apply_all
        else:
            do_reverb = self.reverb_prob > 0 and rng.random() < self.reverb_prob
            do_demand = self.demand_prob > 0 and bool(self._tracks) and rng.random() < self.demand_prob
            do_bandwidth = self.bandwidth_prob > 0 and rng.random() < self.bandwidth_prob

        if do_reverb:
            audio = self._apply_reverb(audio, rng)

        if do_demand:
            audio = self._apply_demand(audio, rng)

        if do_bandwidth:
            audio = self._apply_bandwidth(audio, rng)

        return torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)

    def _apply_reverb(self, audio, rng):

        try:
            import pyroomacoustics as pra
        except ImportError as exc:
            raise ImportError(
                "HardArtificialNoise's reverb stage requires pyroomacoustics (pip install pyroomacoustics). "
                "Set reverb_prob=0 to disable it if you don't need reverb."
            ) from exc

        room_length = float(rng.uniform(*self.reverb_length_range))
        room_width = float(rng.uniform(*self.reverb_width_range))
        room_height = float(rng.uniform(*self.reverb_height_range))
        rt60_target = float(rng.uniform(*self.reverb_rt60_range))
        room_dim = [room_length, room_width, room_height]

        try:
            e_absorption, max_order = pra.inverse_sabine(rt60_target, room_dim)
        except ValueError:
            e_absorption, max_order = 0.2, 10

        room = pra.ShoeBox(room_dim, fs=self.sample_rate, materials=pra.Material(e_absorption), max_order=max_order)
        source_pos = [room_length * 0.4, room_width * 0.5, room_height * 0.5]
        mic_pos = [room_length * 0.6, room_width * 0.5, room_height * 0.5]
        room.add_source(source_pos, signal=audio)
        room.add_microphone(np.array(mic_pos).reshape(3, 1))
        room.compute_rir()
        room.simulate()

        reverb_audio = room.mic_array.signals[0].astype(np.float32)
        reverb_audio = self._align_length(reverb_audio, audio.shape[0])
        return self._match_rms(reverb_audio, audio)

    def _apply_demand(self, audio, rng):

        noise_audio = self._tracks[int(rng.integers(0, len(self._tracks)))]
        noise_audio = self._fit_length(noise_audio, audio.shape[0], rng)
        snr_db = float(rng.uniform(*self.demand_snr_db_range))
        return self._mix_at_snr(audio, noise_audio, snr_db)

    def _apply_bandwidth(self, audio, rng):

        try:
            import scipy.signal as signal
        except ImportError as exc:
            raise ImportError("HardArtificialNoise's bandwidth stage requires scipy (pip install scipy).") from exc

        chosen_filter = self.bandwidth_filters[int(rng.integers(0, len(self.bandwidth_filters)))]
        chosen_cutoff = self.bandwidth_cutoffs_hz[int(rng.integers(0, len(self.bandwidth_cutoffs_hz)))]

        nyquist = 0.5 * self.sample_rate
        chosen_cutoff = min(chosen_cutoff, nyquist - 100)
        normalized_cutoff = chosen_cutoff / nyquist

        if chosen_filter == "butter":
            sos = signal.butter(self.bandwidth_filter_order, normalized_cutoff, btype="low", output="sos")
        elif chosen_filter == "bessel":
            sos = signal.bessel(self.bandwidth_filter_order, normalized_cutoff, btype="low", norm="mag", output="sos")
        elif chosen_filter == "cheby1":
            sos = signal.cheby1(self.bandwidth_filter_order, rp=1.0, Wn=normalized_cutoff, btype="low", output="sos")
        else:
            raise ValueError(f"Unsupported bandwidth filter type: {chosen_filter}")

        filtered_audio = signal.sosfilt(sos, audio).astype(np.float32)
        return self._match_rms(filtered_audio, audio)

    @staticmethod
    def _fit_length(audio, target_length, rng):
        if audio.shape[0] == target_length:
            return audio
        if audio.shape[0] > target_length:
            start = int(rng.integers(0, audio.shape[0] - target_length + 1))
            return audio[start:start + target_length]
        repeats = int(np.ceil(target_length / audio.shape[0]))
        return np.tile(audio, repeats)[:target_length]

    @staticmethod
    def _align_length(audio, target_length):
        if audio.shape[0] >= target_length:
            return audio[:target_length]
        return np.pad(audio, (0, target_length - audio.shape[0]))

    @staticmethod
    def _match_rms(audio, reference):
        audio_power = np.mean(np.square(audio))
        reference_power = np.mean(np.square(reference))
        if audio_power <= 0 or reference_power <= 0:
            return audio.astype(np.float32)
        return (audio * np.sqrt(reference_power / audio_power)).astype(np.float32)

    @staticmethod
    def _mix_at_snr(audio, noise, snr_db):
        speech_power = np.mean(np.square(audio))
        noise_power = np.mean(np.square(noise))
        if speech_power <= 0 or noise_power <= 0:
            return audio
        target_noise_power = speech_power / (10 ** (snr_db / 10.0))
        scale = np.sqrt(target_noise_power / noise_power)
        mixed = audio + scale * noise
        peak = np.max(np.abs(mixed))
        if peak > 1.0:
            mixed = mixed / peak * 0.99
        return mixed.astype(np.float32)

def align_video_to_audio(video, audio):

    Tv, H, W, C = video.shape # (Tv, H, W, Cv)
    Ta = audio.shape[0] # (Ta,)

    target_Tv = Ta // (160*2*2) + 1
    padding = target_Tv - Tv

    if padding >= 0:
        padding_left = padding // 2
        padding_right = padding - padding_left
        video = torch.cat([video.new_zeros(padding_left, H, W, C), video, video.new_zeros(padding_right, H, W, C)], dim=0)
    else:
        # Video has more frames than the audio-derived target length. This happens
        # with independently-trimmed video/audio segments (e.g. manifest-built
        # datasets) even when both are correctly 25fps/16kHz -- a few tens of ms of
        # boundary jitter is enough to tip Tv one frame over target_Tv. Crop
        # symmetrically instead of padding with a negative amount.
        crop = -padding
        crop_left = crop // 2
        crop_right = crop - crop_left
        video = video[crop_left: Tv - crop_right]

    return video

class LipDetectCrop(nn.Module):
    
    def __init__(
        self, 
        mean_face_landmarks_path="datasets/LRS3/20words_mean_face.npy",
        start_idx=48,
        stop_idx = 68,
        crop_width = 96,
        crop_height = 96,
        window_margin = 12,
        STD_SIZE = (256, 256),
        stablePntsIDs = [33, 36, 39, 42, 45]
    ):
        super().__init__()
        
        self.mean_face_landmarks = np.load(mean_face_landmarks_path)
        self.start_idx = start_idx
        self.stop_idx = stop_idx
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.window_margin = window_margin
        self.STD_SIZE = STD_SIZE
        self.stablePntsIDs = stablePntsIDs
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            self.face_detector = RetinaFacePredictor(device=device, threshold=0.8, model=RetinaFacePredictor.get_model('resnet50'))
            self.landmark_detector = FANPredictor(device=device, model=None)
        except:
            pass
        
    def detect_landmarks(self, video, verbose=0):
        
        video_landmarks = []

        for i, frame in enumerate(video):
            if verbose:
                sys.stdout.write("\r{}/{}".format(i+1, video.shape[0]))
            frame = frame.numpy()
            detected_faces = self.face_detector(frame, rgb=True)
            landmarks, scores = self.landmark_detector(frame, detected_faces, rgb=True)
            if len(landmarks) > 0:
                video_landmarks.append(landmarks[0])
            else:
                video_landmarks.append(None)
            
        return video_landmarks

    def landmarks_interpolate(self, landmarks):
            """landmarks_interpolate.
            :param landmarks: List, the raw landmark (in-place)
            """
            valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
            if not valid_frames_idx:
                return None
            for idx in range(1, len(valid_frames_idx)):
                if valid_frames_idx[idx] - valid_frames_idx[idx-1] == 1:
                    continue
                else:
                    landmarks = self.linear_interpolate(landmarks, valid_frames_idx[idx-1], valid_frames_idx[idx])
            valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
            # -- Corner case: keep frames at the beginning or at the end failed to be detected.
            if valid_frames_idx:
                landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
                landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
            valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
            assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
            return landmarks

    def linear_interpolate(self, landmarks, start_idx, stop_idx):
            """linear_interpolate.
            :param landmarks: ndarray, input landmarks to be interpolated.
            :param start_idx: int, the start index for linear interpolation.
            :param stop_idx: int, the stop for linear interpolation.
            """
            start_landmarks = landmarks[start_idx]
            stop_landmarks = landmarks[stop_idx]
            delta = stop_landmarks - start_landmarks
            for idx in range(1, stop_idx-start_idx):
                landmarks[start_idx+idx] = start_landmarks + idx/float(stop_idx-start_idx) * delta
            return landmarks

    def warp_img(self, src, dst, img, std_size):
            """warp_img.
            :param src: ndarray, source coordinates.
            :param dst: ndarray, destination coordinates. 
            :param img: ndarray, an input image.
            :param std_size: tuple (rows, cols), shape of the output image generated.
            """
            tform = skimage.transform.estimate_transform('similarity', src, dst)  # find the transformation matrix
            warped = skimage.transform.warp(img, inverse_map=tform.inverse, output_shape=std_size)  # wrap the frame image
            warped = warped * 255  # note output from wrap is double image (value range [0,1])
            warped = warped.astype('uint8')
            return warped, tform

    def apply_transform(self, transform, img, std_size):
            """apply_transform.
            :param transform: Transform object, containing the transformation parameters \
                            and providing access to forward and inverse transformation functions.
            :param img: ndarray, an input image.
            :param std_size: tuple (rows, cols), shape of the output image generated.
            """
            warped = skimage.transform.warp(img, inverse_map=transform.inverse, output_shape=std_size)
            warped = warped * 255  # note output from wrap is double image (value range [0,1])
            warped = warped.astype('uint8')
            return warped

    def cut_patch(self, img, landmarks, height, width, threshold=5):
            """cut_patch.
            :param img: ndarray, an input image.
            :param landmarks: ndarray, the corresponding landmarks for the input image.
            :param height: int, the distance from the centre to the side of of a bounding box.
            :param width: int, the distance from the centre to the side of of a bounding box.
            :param threshold: int, the threshold from the centre of a bounding box to the side of image.
            """
            center_x, center_y = np.mean(landmarks, axis=0)

            if center_y - height < 0:                                                
                center_y = height                                                    
            if center_y - height < 0 - threshold:                                    
                raise Exception('too much bias in height')                           
            if center_x - width < 0:                                                 
                center_x = width                                                     
            if center_x - width < 0 - threshold:                                     
                raise Exception('too much bias in width')                            
                                                                                    
            if center_y + height > img.shape[0]:                                     
                center_y = img.shape[0] - height                                     
            if center_y + height > img.shape[0] + threshold:                         
                raise Exception('too much bias in height')                           
            if center_x + width > img.shape[1]:                                      
                center_x = img.shape[1] - width                                      
            if center_x + width > img.shape[1] + threshold:                          
                raise Exception('too much bias in width')                            
                                                                                    
            cutted_img = np.copy(img[ int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                                int(round(center_x) - round(width)): int(round(center_x) + round(width))])
            return cutted_img

    def crop_patch(self, video, landmarks):

            """Crop mouth patch
            :param str video_pathname: pathname for the video_dieo
            :param list landmarks: interpolated landmarks
            """

            frame_idx = 0
            num_frames = video.shape[0]
            frame_gen = video
            margin = min(num_frames, self.window_margin)
            while True:
                try:
                    #frame = frame_gen.__next__() ## -- BGR
                    frame = frame_gen[frame_idx]
                except StopIteration:
                    break
                if frame_idx == 0:
                    q_frame, q_landmarks = collections.deque(), collections.deque()
                    sequence = []

                q_landmarks.append(landmarks[frame_idx])
                q_frame.append(frame)
                if len(q_frame) == margin:
                    smoothed_landmarks = np.mean(q_landmarks, axis=0)
                    cur_landmarks = q_landmarks.popleft()
                    cur_frame = q_frame.popleft()

                    # -- affine transformation
                    trans_frame, trans = self.warp_img( smoothed_landmarks[self.stablePntsIDs, :],
                                                self.mean_face_landmarks[self.stablePntsIDs, :],
                                                cur_frame,
                                                self.STD_SIZE)
                    trans_landmarks = trans(cur_landmarks)

                    # -- crop mouth patch
                    sequence.append( self.cut_patch( trans_frame,
                                                trans_landmarks[self.start_idx:self.stop_idx],
                                                self.crop_height//2,
                                                self.crop_width//2,))
                if frame_idx == len(landmarks)-1:
                    while q_frame:

                        cur_frame = q_frame.popleft()

                        # -- transform frame
                        trans_frame = self.apply_transform( trans, cur_frame, self.STD_SIZE)

                        # -- transform landmarks
                        trans_landmarks = trans(q_landmarks.popleft())

                        # -- crop mouth patch
                        sequence.append( self.cut_patch( trans_frame,
                                                    trans_landmarks[self.start_idx:self.stop_idx],
                                                    self.crop_height//2,
                                                    self.crop_width//2,))
                    return np.array(sequence)
                frame_idx += 1
            return None
        
    def forward(self, video, verbose=0):
        
        landmarks = self.detect_landmarks(video, verbose=verbose)
        preprocessed_landmarks = self.landmarks_interpolate(landmarks)
        video_crop = self.crop_patch(video.numpy(), preprocessed_landmarks)
        video_crop = torch.tensor(video_crop)
        
        return video_crop