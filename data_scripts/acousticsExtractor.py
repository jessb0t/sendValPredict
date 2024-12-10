import sys
import os
import subprocess
import audiofile
import opensmile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


### SETUP
inpath = "/Users/jessicaalexander/github/sendValPredict/data/wavs/"
outpath = "/Users/jessicaalexander/github/sendValPredict/data/egemaps/"

#collect all files
allwavs = [f for f in os.listdir(inpath) if f.endswith('.wav')]

#feature extractor
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

for i in range(len(allwavs)):
    file_in = os.path.join(inpath, allwavs[i])
    filename = allwavs[i][:-4]
    print(filename)
    
    #read in file
    signal, sr = audiofile.read(file_in)
    signal_mono = np.mean(signal, 0) #average from stereo to mono
    # X = range(len(signal_mono))
    # plt.plot(X, signal_mono)
    # plt.show()
    
    #loop over 5 second windows within a file (always dropping the last, incomplete window)
    samplepoints_total = np.shape(signal_mono)[0]
    samplePoints_per_window = 5*sr
    numWindows = int(np.ceil(samplepoints_total/samplePoints_per_window))
    timesteps = np.arange(0, (numWindows * samplePoints_per_window), samplePoints_per_window)
    idxStart = 0
    idxStop = 1
    for j in range(numWindows):
        windowStart = int(timesteps[idxStart])
        if j < numWindows-1:
            windowEnd = int(timesteps[idxStop])
        else:
            windowEnd = samplepoints_total
        window = slice(windowStart, windowEnd)
        signal_sliced = signal_mono[window]
        
        #extract features and concatenate into dataframe
        features = smile.process_signal(signal_sliced, sr)
        features.insert(0, "StartIdx", [idxStart*samplePoints_per_window/sr], True)
        features.insert(1, "StopIdx", [idxStop*samplePoints_per_window/sr], True)
        if idxStart==0:
            df = features
        else:
            df = pd.concat([df, features])
        idxStart += 1
        idxStop += 1
    
    #track the number of windows, per feature, that could not be extracted (output as '0') for this file
    if i == 0:
        dfZeros = sum(df.values==0)
    else:
        newrow = sum(df.values==0)
        dfZeros = np.vstack([dfZeros, newrow])
    
    #output extracted features as csv
    fileparts = filename.split("_")
    spkr = fileparts[0] #identify speaker ID
    video = fileparts[1] #identify video ID
    file_out_fea = os.path.join(outpath, spkr + "_" + video + "_egemaps.csv")
    df.to_csv(file_out_fea)

#output tracking csv for number of un-extractable windows, per feature, per file
#dfZeros = pd.DataFrame(dfZeros)
#dfZeros.columns = list(df.columns)
#dfZeros.to_csv(os.path.join(outpath, "zeros.csv"))


## confirm sampling rate across dataset
# allwavs = [w for w in os.listdir(inpath) if w.endswith('.wav')]
# sampling_rates = np.zeros(len(allwavs))
# for index, file in enumerate(allwavs):
#     f = os.path.join(inpath, file)
#     signal, sr = audiofile.read(f)
#     sampling_rates[index] = sr