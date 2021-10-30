'''
This script is used to convert raw audio data into mel-spectrograms for image processing and 
computer vision analysis of audio files. The script is based on EmoDB dataset, which can be found 
at: https://www.kaggle.com/piyushagni5/berlin-database-of-emotional-speech-emodb
'''

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display
from scipy.io import wavfile

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--audio_dir',type=str, help='path to directory containing audio files per emotion class')
parser.add_argument('--data_dir', type=str, default=os.getcwd(), help='path where images will be stored')
args = parser.parse_args()

AUDIO_DIR = args.audio_dir
IMG_DIR = args.data_dir

# save audios as spectrograms
for AUDIO_FILE in os.listdir(AUDIO_DIR):
    samples, sample_rate = librosa.load(os.path.join(AUDIO_DIR, AUDIO_FILE), sr=None)
    sgram = librosa.stft(samples)
    sgram_mag, _ = librosa.magphase(sgram)
    mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
    mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
    librosa.display.specshow(mel_sgram, sr=sample_rate, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB').remove()
    plt.axis('off')
    plt.savefig(f"./{IMG_DIR}/{AUDIO_FILE}.png")

### REPLACE THE LABEL:EMOTION DICTIONARY WITH THAT RELEVANT TO DATASET ###
label2name = {
    "L": "Boredom",
    "A": "Fear",
    "E": "Disgust",
    "F": "Happiness",
    "T": "Sadness",
    "W": "Anger",
    "N": "Neutral"
}

# move the spectrograms to respective class folders   
for cat in label2name.keys():
    os.mkdir(os.path.join(IMG_DIR, label2name[cat]))

for filename in os.listdir(IMG_DIR):
    if os.path.isfile(IMG_DIR+"/"+filename) and filename[:-3] == "png":
        cat = filename[5]
        shutil.move(os.path.join(IMG_DIR, filename), os.path.join(IMG_DIR, label2name[cat]))

print("====> All audios converted to mel-spectrograms !!")




