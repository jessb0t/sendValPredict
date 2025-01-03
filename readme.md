## Predicting Human Valence Ratings of Natural Spoken Narratives from Lexical and Acoustic Information

### Overview
The cues used by human observers to predict the emotional state of an interlocutor are ambiguous, and this is particularly true for vocal emotion ([Atias et al., 2019](https://doi.apa.org/doi/10.1037/xge0000535)).  Efforts to predict the negative-to-positive degree of “pleasantness” of emotional speech (termed “valence”) have been especially fraught ([Busso & Rahman, 2012](https://www.isca-speech.org/archive/interspeech_2012/busso12_interspeech.html)).  According to prior work using time-series models ([Ong et al., 2021](https://ieeexplore.ieee.org/document/8913483/)), the perception of speaker valence by human listeners in an auditory-only modality is predominantly based on signal semantics, with acoustic features such as prosodic contour and voice quality demonstrating weaker explanatory power.  Here, I investigate several linear regression models to compare the explanatory power of semantic and acoustic information, both alone and in combination.  Furthermore, I explore the extent to which a [fine-tuned, self-supervised transformer model](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim) is able to simulate human behavior in valence ratings of natural, spoken narratives.

This project analyzes data provided in the first release the [Stanford Emotional Narratives Dataset](https://github.com/StanfordSocialNeuroscienceLab/SEND).  For results of analyses to date, please see the `results` folder of this repository, which includes results both as a readme (for easy online viewing), as well as a .docx file.

Repository License: CC BY-SA 4.0

### Scripts

#### Data Extraction and Organization

##### `acousticsExtractor.py`

_inputs:_ audio files in .wav format

_outputs:_ for each input .wav, a .csv file with 88 extracted acoustic features ([eGeMAPS](https://audeering.github.io/opensmile-python/)) for every five-second window in the file {data/egemaps}

##### `dataCompiler.py`

_inputs:_

* time-aligned transcription files from the SEND dataset
* time-aligned, aggregated (Evaluator Weighted Estimator) human valence ratings from the SEND dataset
* acoustic features extracted (via `acousticsExtractor.py`) from .wav files exported from SEND videos
* lexical valence and arousal norms collected by [Warriner et al. (2013)](https://link.springer.com/article/10.3758/s13428-012-0314-x)

_outputs:_ dataframe with each row representing a five-second window and columns providing data relating to the identification, human rating, lexical/semantic features, and acoustic features of the window  {"data.csv"}

##### `audeeringPredictions.py`

_inputs:_

* audio files in .wav format
* data.csv

_outputs:_ dataframe with each row representing a five-second window and columns providing data relating to the identification, human rating, and model-predicted rating of the window  {"llm_prediction_data.csv"}

#### Modeling

##### `analysisRegressions.R`

_inputs:_ data.csv

_outputs:_ for four models, a plot comparing the human gold standard valence rating for each five-second window and the valence rating predicted by the model, including the coefficient of determination, Pearson correlation coefficient, and concordance correlation coefficient; the script also contains in-line code to explore the beta weights for statistically significant predictors in each model {figs}

##### `analysisLLM.R`

_inputs:_ llm_prediction_data.csv

_outputs:_ a plot comparing the human gold standard valence rating for each five-second window and the valence rating predicted by the model, including the coefficient of determination, Pearson correlation coefficient, and concordance correlation coefficient {figs}
