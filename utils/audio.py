# Adapted from KoelLabs github
import numpy as np
import librosa
import soundfile as sf
import ffmpeg
import scipy.io.wavfile as wavfile
from io import BytesIO

WAV_HEADER_SIZE = 44
TARGET_SAMPLE_RATE = 16000


def calculate_wps(num_words: int, audio_length: int, sr=16000):
    if audio_length <= 0 or num_words <= 0:
        return 0.0
    seconds = audio_length / sr
    wps = num_words / seconds
    return wps


def pitch_average(audio_data, sr=16000):
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32) / 32768.0
    pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)

    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)
    assert len(pitch_values) > 0, "No pitch values found in the audio data"
    return float(np.mean(pitch_values))


def audio_resample(array, src_sample_rate, target_sample_rate=TARGET_SAMPLE_RATE):
    if src_sample_rate == target_sample_rate:
        return array
    return np.interp(
        np.linspace(
            0,
            len(array),
            int(len(array) * target_sample_rate / src_sample_rate),
        ),
        np.arange(len(array)),
        array,
    ).astype(np.int16)


def audio_bytes_to_array(
    data,
    src_sample_rate=None,
    target_sample_rate=TARGET_SAMPLE_RATE,
    output_orig_sample_rate=False,
):
    # TODO: rename to make clear this requires WAV format
    assert data[:4] == b"RIFF", "Not a WAV file, first 4 bytes are not RIFF: " + data[
        :4
    ].decode("utf-8")
    if src_sample_rate == None:
        # read 32 bit integer from bytes 25-28 in header
        src_sample_rate = int.from_bytes(data[24:28], byteorder="little")
    # read bits per sample from bytes 35-36 in header
    bits_per_sample = int.from_bytes(data[34:36], byteorder="little")
    dtype = np.int16 if bits_per_sample == 16 else np.int32
    # read number of channels from bytes 23-24 in header
    num_channels = int.from_bytes(data[22:24], byteorder="little")
    data = data[WAV_HEADER_SIZE:]
    audio = np.frombuffer(data, dtype=dtype).astype(np.int16)
    # average in chunks of num_channels
    if num_channels > 1:
        if len(audio) % num_channels != 0:
            audio = audio[: -(len(audio) % num_channels)]
        audio = audio.reshape(-1, num_channels)
        audio = np.mean(audio, axis=1).astype(np.int16)
    audio = audio_resample(audio, src_sample_rate, target_sample_rate)
    if output_orig_sample_rate:
        return audio, src_sample_rate
    return audio


def audio_bytes_to_wav_array(
    bytes,
    format,
    output_sample_rate=TARGET_SAMPLE_RATE,
    output_orig_sample_rate=False,
):
    wav_bytes = (
        ffmpeg.input("pipe:0", format=format)
        .output("pipe:1", format="wav")
        .run(input=bytes, capture_stdout=True, capture_stderr=True)
    )
    return audio_bytes_to_array(
        wav_bytes[0],
        target_sample_rate=output_sample_rate,
        output_orig_sample_rate=output_orig_sample_rate,
    )


def m4a_file_to_wav(input_path, output_path, sample_rate=TARGET_SAMPLE_RATE):
    ffmpeg.input(input_path).output(
        output_path,
        format="wav",
        ar=sample_rate,
        ac=1,  # Audio channels (1 for mono)
    ).overwrite_output().run(quiet=True)

    return output_path


def qta_to_wav(input_path, output_path, sample_rate=16000):
    # We explicitly map only the AAC stream to avoid the APAC decode error.
    audio = ffmpeg.input(input_path)["a:0"]
    ffmpeg.output(
        audio, output_path, format="wav", ar=sample_rate, ac=1
    ).overwrite_output().run(quiet=True)
    return output_path


def audio_file_to_wav(input_path, output_path, sample_rate=TARGET_SAMPLE_RATE):
    ext = input_path.rsplit(".", 1)[-1].lower()
    probe = ffmpeg.probe(
        input_path
    )  # Probe actual format, files are sometimes mislabeled

    true_format = probe["format"]["format_name"].split(",")[0]
    if ext == "qta":
        qta_to_wav(input_path, output_path, sample_rate)
    elif true_format in ("mov", "mp4", "m4a"):
        m4a_file_to_wav(input_path, output_path, sample_rate)
    else:
        # mp3, ogg, aac, flac, opus, mpeg — decode via ffmpeg bytes pipeline then resample
        with open(input_path, "rb") as f:
            audio_bytes = f.read()
        array = audio_bytes_to_wav_array(audio_bytes, format=true_format)
        sf.write(output_path, array, sample_rate)
    return output_path


def audio_file_to_array(
    input_path, desired_sample_rate=TARGET_SAMPLE_RATE, output_orig_sample_rate=False
):
    rate, data = wavfile.read(input_path)
    data = audio_dual_channel_to_mono(data)
    data = audio_resample(data, rate, desired_sample_rate)
    if output_orig_sample_rate:
        return data, rate
    return data


def audio_array_to_bytes(array, sample_rate=TARGET_SAMPLE_RATE):
    with BytesIO() as f:
        wavfile.write(f, sample_rate, array)
        return f.getvalue()


def audio_dual_channel_to_mono(input_array):
    if input_array.ndim == 2 and input_array.shape[1] == 2:
        return np.mean(input_array, axis=1).astype(np.int16)
    return input_array
