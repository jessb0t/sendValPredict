import os
from scipy.io import wavfile
import resampy
import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
import pandas as pd

##################################################
#https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim

class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x

class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits

# load model from hub
device = 'cpu'
model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = EmotionModel.from_pretrained(model_name).to(device)

# dummy signal
#sampling_rate = 16000
#signal = np.zeros((1, sampling_rate), dtype=np.float32)

def process_func(
    x: np.ndarray,
    sampling_rate: int,
    embeddings: bool = False,
) -> np.ndarray:
    r"""Predict emotions or extract embeddings from raw audio signal."""

    # run through processor to normalize signal
    # always returns a batch, so we just get the first entry
    # then we put it on the device
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = y.reshape(1, -1)
    y = torch.from_numpy(y).to(device)

    # run through model
    with torch.no_grad():
        y = model(y)[0 if embeddings else 1]

    # convert to numpy
    y = y.detach().cpu().numpy()

    return y

#print(process_func(signal, sampling_rate))
#  Arousal    dominance valence
# [[0.5460754  0.6062266  0.40431657]]

#print(process_func(signal, sampling_rate, embeddings=True))
# Pooled hidden states of last transformer layer
# [[-0.00752167  0.0065819  -0.00746342 ...  0.00663632  0.00848748
#    0.00599211]]

##################################################

#import dataframe for identification of time windows and trials to drop
df = pd.read_csv("data.csv")

#establish target sampling rate
SR = 16000

#identify all wav files
inpath = "/Users/jessicaalexander/github/sendValPredict/data/wavs/"
allwavs = [f for f in os.listdir(inpath) if f.endswith('.wav')]

#set up dictionary to collect model predictions
my_dict = {"spkr": [], "vid": [], "time": [], "ewe": [], "valpred": []}

#loop over each wav file
for i, w in enumerate(allwavs):
    file_in = os.path.join(inpath, w)
    filename = w[:-4]
    print(filename)
    print(i)
    fileparts = filename.split("_")
    spkr = fileparts[0] #identify speaker ID
    video = fileparts[1] #identify video ID

    #read in wav file, convert to mono, downsample
    sr, signal = wavfile.read(file_in)
    signal_mono = np.mean(signal, 1) #average from stereo to mono
    signal_mono_downsampled = resampy.resample(signal_mono, sr, SR) #downsample to 16kHz to match LM requirements

    #grab each window if it is in the dataframe and is not the first five seconds of the video
    dfsubset = df[(df.spkr==spkr) & (df.video==video) & (df.time>0)] 
    for r in range(dfsubset.shape[0]):
        rowdat = dfsubset.iloc[[r], list(range(5,dfsubset.shape[1]))]
        ewe = rowdat["ewe"].values.item() #grab ground truth
        tmp = np.isfinite(rowdat.values)
        nonfinites = np.sum(tmp==False)
        if nonfinites > 0: #drop windows with non-finite values
            continue
        else:
            fivesecs = SR*5
            time = dfsubset.iloc[[r], [3]]
            time = time.values[0][0]
            time = pd.to_numeric(time)
            print(time)
            start = time * SR
            end = start + fivesecs
            input_sample = signal_mono_downsampled[start:end] #carve off five seconds of audio

            output = process_func(input_sample, SR) #process audio through model
            val = output[:,2]
            valpred = val.item() #grab valence prediction

            my_dict["spkr"].append(spkr)
            my_dict["vid"].append(video)
            my_dict["time"].append(time)
            my_dict["ewe"].append(ewe/100) #convert to range 0-1
            my_dict["valpred"].append(valpred)

out = pd.DataFrame.from_dict(my_dict)
out.to_csv("llm_prediction_data.csv")
