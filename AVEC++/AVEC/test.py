import torchvision, torchaudio
uid = "_vYdDrMHC0U_00001"
video, _, info = torchvision.io.read_video(f"datasets/LRS2/mvlrs_v1/main/{uid}_mouth.mp4", pts_unit="sec")
audio, sr = torchaudio.load(f"datasets/LRS2/mvlrs_v1/main/{uid}.flac")
print("video fps:", info["video_fps"], "frames:", video.shape[0], "video_dur_s:", video.shape[0]/info["video_fps"])
print("audio sr:", sr, "samples:", audio.shape[1], "audio_dur_s:", audio.shape[1]/sr)