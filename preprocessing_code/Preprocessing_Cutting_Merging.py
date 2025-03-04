import pandas as pd
import os
import wave
import contextlib
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.io import wavfile
import librosa
import soundfile as sf


def wav_infos(wav_path):
    """
    Getting audio information

    :param wav_path: audio path
    :return: [1, 2, 8000, 51158, 'NONE', 'not compressed']
    Correspondence: channel, sample width, frame rate, number of frames, unique identifier, lossless
    """
    with wave.open(wav_path, "rb") as f:
        return list(f.getparams())


def read_wav(wav_path):
    """
    Read audio file content: can only read mono audio files, this is more time-consuming

    :param wav_path: audio path
    :return: audio content
    """
    with wave.open(wav_path, "rb") as f:
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        # Read the sound data, passing a parameter specifying the length (in sampling points) to be read
        str_data = f.readframes(nframes)
    return str_data


def get_wav_time(wav_path):
    """
    Get the audio file is duration

    :param wav_path: audio path
    :return: audio duration (in seconds)
    """
    with contextlib.closing(wave.open(wav_path, "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return duration


def get_ms_part_wav(main_wav_path, start_time, end_time, part_wav_path):
    """
    Audio slicing to get part of the audio in milliseconds

    :param main_wav_path: path of the original audio file
    :param start_time: the start time of the capture in milliseconds
    :param end_time: end time of the capture in milliseconds
    :param part_wav_path: the path of the intercepted audio file
    :return.
    """
    # Convert milliseconds to seconds for librosa
    start_time_sec = int(start_time) / 1000
    end_time_sec = int(end_time) / 1000

    # Load audio file
    y, sr = librosa.load(main_wav_path, sr=None)

    # Convert time to samples
    start_sample = int(start_time_sec * sr)
    end_sample = int(end_time_sec * sr)

    # Slice the audio
    y_segment = y[start_sample:end_sample]

    # Write the sliced audio to file
    sf.write(part_wav_path, y_segment, sr)


def get_second_part_wav(main_wav_path, start_time, end_time, part_wav_path):
    """
    Audio slicing to get a portion of the audio in seconds.

    :param main_wav_path: path of the original audio file
    :param start_time: the start time of the capture in seconds
    :param end_time: the end time of the intercept in seconds
    :param part_wav_path: path of the audio file after capture
    :return.
    """
    # Load audio file with scipy.io.wavfile
    sr, data = wavfile.read(main_wav_path)

    # Convert to seconds
    start_time_sec = int(float(start_time))
    end_time_sec = int(float(end_time))

    # Convert time to samples
    start_sample = int(start_time_sec * sr)
    end_sample = int(end_time_sec * sr)

    # Make sure we don't exceed the audio length
    end_sample = min(end_sample, len(data))

    # Slice the audio
    y_segment = data[start_sample:end_sample]

    # Write the sliced audio to file
    wavfile.write(part_wav_path, sr, y_segment)


def get_minute_part_wav(main_wav_path, start_time, end_time, part_wav_path):
    """
    Audio slice to get part of the audio minutes:seconds time style: "12:35"

    :param main_wav_path: path to the original audio file
    :param start_time: the start time of the capture in format "MM:SS"
    :param end_time: end time of the capture in format "MM:SS"
    :param part_wav_path: path to the intercepted audio file
    :return.
    """
    # Convert time format to seconds
    start_time_sec = int(start_time.split(":")[0]) * 60 + int(start_time.split(":")[1])
    end_time_sec = int(end_time.split(":")[0]) * 60 + int(end_time.split(":")[1])

    # Load audio file
    y, sr = librosa.load(main_wav_path, sr=None)

    # Convert time to samples
    start_sample = int(start_time_sec * sr)
    end_sample = int(end_time_sec * sr)

    # Slice the audio
    y_segment = y[start_sample:end_sample]

    # Write the sliced audio to file
    sf.write(part_wav_path, y_segment, sr)


def wav_to_pcm(wav_path, pcm_path):
    """
    Convert wav file to pcm file

    :param wav_path:wav file path
    :param pcm_path: path to the pcm file to be stored
    :return: return result
    """
    f = open(wav_path, "rb")
    f.seek(0)
    f.read(44)

    data = np.fromfile(f, dtype=np.int16)
    data.tofile(pcm_path)
    f.close()


def pcm_to_wav(pcm_path, wav_path):
    """
    pcm file to wav file

    :param pcm_path: pcm file path
    :param wav_path: wav file path
    :return.
    """
    f = open(pcm_path, "rb")
    str_data = f.read()
    wave_out = wave.open(wav_path, "wb")
    wave_out.setnchannels(1)
    wave_out.setsampwidth(2)
    wave_out.setframerate(8000)
    wave_out.writeframes(str_data)
    f.close()
    wave_out.close()


def wav_waveform(wave_path):
    """
    Waveforms corresponding to audio
    :param wave_path:  audio path
    :return:
    """
    y, sr = librosa.load(wave_path, sr=None)
    time = np.arange(0, len(y)) / sr

    plt.figure(figsize=(10, 4))
    plt.plot(time, y, "blue")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    plt.show()


def wav_combine(files_list):
    """
    Combine multiple wav files into one

    :param files_list: list containing [number_of_files, file1, file2, ..., output_path]
    :return: None
    """
    n = files_list[0]  # Number of files to splice
    output_path = files_list[n + 1]  # Output path is the last item

    # Initialize empty array for combined audio
    combined_audio = np.array([])
    sample_rate = None

    # Load and combine each audio file
    for i in range(1, n + 1):
        y, sr = librosa.load(files_list[i], sr=None)

        # Store sample rate from first file
        if sample_rate is None:
            sample_rate = sr

        # Append audio data
        combined_audio = np.append(combined_audio, y)

    # Write combined audio to file
    sf.write(output_path, combined_audio, sample_rate)


def pdfFilesPath(path):
    filePaths = []  # Name of all files in the storage directory with paths
    for root, dirs, files in os.walk(path):
        for file in files:
            filePaths.append(os.path.join(root, file))
    return filePaths


if __name__ == "__main__":
    path = r"./all_data/"  # Audio directories to be cut
    # The cut audio directory, which is also the audio directory to be merged.
    path_segment = r"./audio_split/"
    path1 = r"./data_time1/"  # # excel file directory
    excel_root = os.listdir(path1)  # excel file directory deposit list

    os.makedirs(path_segment, exist_ok=True)
    os.makedirs("./audio_combine/", exist_ok=True)

    a_l = []  # Record the number of audio cuts
    print("Start cutting the audio!")

    for root, dir, files in os.walk(path):
        for i in range(len(files)):
            audio = os.path.join(root, files[i])
            time_all = int(get_wav_time(audio) * 1000)

            print("The audio is %s, the file is %s" % (files[i], excel_root[i]))

            df = pd.read_excel(
                os.path.join(path1, excel_root[i]), usecols=["start_time", "stop_time"]
            )

            index = 1  # Cut serial number names, starting with serial number 1
            k = 0
            # k_l = 0  # Total number of audio cuts recorded
            for j in range(len(df)):
                l = len(df)
                # print("l= %d" % l)
                while k <= l - 1:
                    start_time = float(df.loc[k, "start_time"])
                    end_time = float(df.loc[k, "stop_time"])

                    audio_segment = os.path.join(
                        path_segment, f"{os.path.splitext(files[i])[0]}_{index}.wav"
                    )
                    get_second_part_wav(audio, start_time, end_time, audio_segment)

                    index += 1
                    k += 1

            a_l.append(k)
            print("Cutting out %d audio" % l)

    print("A total of %d audio cuts" % sum(a_l))
    print(a_l)
    print("Audio cutting complete!")

    filePaths = []
    r = []
    for root, dir, files in os.walk(path_segment):
        for file in files:
            f1 = re.findall(r"\d+\.?\d*", file)
            if len(f1) >= 2:  # Check if there are at least 2 numbers in the filename
                f1 = (f1[0], f1[1][:-1] if f1[1][-1:].isalpha() else f1[1])
                f1 = list(map(int, f1))
                r.append(f1)

        if r:  # Only proceed if r has elements
            sorted_r = sorted(r, key=lambda x: (x[0], x[1]))
            new_files = [
                f"{str(sorted_r[i][0])}_AUDIO_{str(sorted_r[i][1])}.wav"
                for i in range(len(sorted_r))
            ]

            for f2 in new_files:
                filePaths.append(os.path.join(root, f2))

    print("*" * 50)
    print("Start merging audio!")

    # Initial value setting
    n = 5  # Select n audios to merge
    id_num = 300  # Starting number of file naming after merger
    w = 1
    # w1,w2 is the range of audio splices to ensure that only files cut from the same audio are merged,
    # avoiding audio splices such as 300 and 301.
    w1 = 0
    w2 = a_l[0] if a_l else 0

    if filePaths:  # Only proceed if there are files to process
        # Process one batch at a time instead of looping through every file
        while w1 < len(filePaths):
            # Make sure w2 doesn't exceed the array bounds
            if w2 > len(filePaths):
                w2 = len(filePaths)

            if w1 >= w2:
                break

            file_1 = filePaths[w1:w2]
            print("Range of w1, w2: %d %d" % (w1, w2))
            id_num2 = 1

            for b in [file_1[i : i + n] for i in range(0, len(file_1), n)]:
                if not b:  # Skip empty batches
                    continue

                out_path = os.path.join("./audio_combine/", f"{id_num}_{id_num2}.wav")
                batch_with_metadata = [len(b)] + b + [out_path]
                wav_combine(batch_with_metadata)
                id_num2 += 1

            if w < len(a_l):
                w1 = w2
                w2 = w2 + a_l[w]
                # Because there are no audio files with the serial numbers 342,394,398,460 in the dataset file, these numbers are skipped.
                if id_num == 341 or id_num == 393 or id_num == 397 or id_num == 459:
                    id_num += 2
                else:
                    id_num += 1
                w += 1
            else:
                break

    print("Audio merge complete!")
