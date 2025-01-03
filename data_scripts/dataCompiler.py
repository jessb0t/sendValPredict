import os
import statistics
import pandas as pd
import numpy as np
import spacy
from spacy.cli import download
import gensim.downloader as gensim_api

## SETUP
#data path
inpath = "/Users/jessicaalexander/github/sendValPredict/data"

#Warriner arousal and valence ratings
warriner_path = "/Users/jessicaalexander/github/sendValPredict/data/resources/BRM-emot-submit_downloaded_2021-08-08.csv"
warriner = pd.read_csv(warriner_path)

#spacy and glove
#download("en_core_web_sm")
spacy_obj = spacy.load("en_core_web_sm")
gspace = gensim_api.load("glove-wiki-gigaword-300")

## FUNCTIONS
#centroid for word embeddings
def find_centroid(sentence, gspace, spacy_obj):
    """
    Function to compute the centroid of the content in a sentence
    by summation of their vectors.
    INPUTS:
    - sentence (string)
    - gspace (word embeddings loaded from gensim)
    - spacy_obj (tokenized spacy model)
    OUTPUT:
    - centroid (300-dimension array)
    """
    processed = spacy_obj(sentence)
    nonstop = [w.lemma_ for w in processed if not w.is_stop and not w.is_punct]
    nonstop_with_vector = [w for w in nonstop if w in gspace]
    
    if len(nonstop_with_vector)>0:
        centroid = sum([gspace[w] for w in nonstop_with_vector])
    else:
        centroid = None
    return centroid

#averaged lexical valence and arousal from Warriner
def compute_val_aro(chunk, warriner, spacy_obj):
    """
    Function to compute: the averaged lexical valence and arousal and
    of lemmas in a chunk based on the Warriner et al. (2013) ratings.
    Ratings are normalized based on the Warriner dataset max of 9,
    such that values range from 0 to 1.
    INPUTS:
    - sentence (string)
    - warriner (dataframe of human ratings)
    - spacy_obj (tokenized spaCy model)
    OUTPUT:
    - valMean (float)
    - aroMean (float)
    """
    processed = spacy_obj(chunk)
    lemmas = [w.lemma_ for w in processed if not w.is_punct]
    lemmas_with_ratings = [w for w in lemmas if w in list(warriner["Word"])]
    
    if len(lemmas_with_ratings)>0:
        ids = [list(warriner["Word"]).index(w) for w in lemmas_with_ratings]
        vals = [list(warriner["V.Mean.Sum"])[idx]/9 for idx in ids]
        aros = [list(warriner["A.Mean.Sum"])[idx]/9 for idx in ids]
        valMean = np.mean(vals)
        aroMean = np.mean(aros)
    else:
        valMean = None
        aroMean = None
    return valMean, aroMean

#loop over all transcription files to build dataframe
all_dfs = []
for i,fileT in enumerate(os.listdir(os.path.join(inpath, "transcriptions"))):
    if fileT.endswith("tsv"):
        print(fileT)
        filepartsT = fileT.split("_")
        spk = filepartsT[0] #identify speaker ID
        vid = filepartsT[1] #identify video ID

        #read in transcription data
        df = pd.read_csv(os.path.join(inpath, "transcriptions", fileT), sep='\t')
        rows = df.shape[0]
        words = list(df['words']) #output: list of words produced in each time window

        #create lexical feature lists
        mean_valence = [] #output: average lexical valence for words produced in each time window
        mean_arousal = [] #output: average lexical arousal for words produced in each time window
        glove = {i: [] for i in range(300)} #output: 300-dimensional summed word vector for words produced in each time window

        for c, chunk in enumerate(words):
            if not pd.isna(chunk):
                valMean, aroMean = compute_val_aro(chunk, warriner, spacy_obj)
                centroid = find_centroid(chunk, gspace, spacy_obj)
                try:
                    tmp = centroid.any()
                    for i, val in enumerate(centroid):
                        glove[i].append(val)
                except:
                    for i in range(300):
                        glove[i].append(None)
            else:
                valMean = None
                aroMean = None
                for i in range(300):
                    glove[i].append(None)
            
            mean_valence.append(valMean)
            mean_arousal.append(aroMean)

            #extract matching acoustic features
            fileA = spk + "_" + vid + "_egemaps.csv"
            egemapsDat = pd.read_csv(os.path.join(inpath, "egemaps", fileA))
            cols = range(4, egemapsDat.shape[1])
            acoustics = egemapsDat[egemapsDat.columns[cols]][0:rows]
            acoustics_dict = acoustics.to_dict('series')

            #find ratings file
            fileR = "results_" + spk[2:] + "_" + vid[3] + ".csv"
            ratingsDat = pd.read_csv(os.path.join(inpath, "ratings_ewe", fileR))
            ratings = []
            for d in range(rows):
                summed = 0
                for n in range(10):
                    sample = n + (d*10)
                    summed += ratingsDat["evaluatorWeightedEstimate"][sample]
                    averaged = summed/10
                ratings.append(averaged)

        spkr = [spk for i in range(rows)] #output: identity of speaker
        video = [vid for i in range(rows)] #output: video ID
        time = list(df['time']) #output: time stamp

        dict = {'spkr': spkr, 'video': video, 'time': time, 'words': words, 'valMean': mean_valence, 'aroMean': mean_arousal, 'ewe': ratings}
        dict = dict | acoustics_dict | glove
        all_dfs.append(dict)

frames = [pd.DataFrame(vid) for vid in all_dfs]
out = pd.concat(frames)

out.to_csv("data.csv")
