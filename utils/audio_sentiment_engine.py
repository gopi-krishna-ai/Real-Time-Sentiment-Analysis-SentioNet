import os,joblib,numpy as np
import librosa
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH=os.path.join(BASE_DIR,'model','audio_sentiment_model.pkl')
_model=None

def _features(path,sr=22050):
    y,rate=librosa.load(path,sr=sr,mono=True)
    if y.size==0: raise ValueError('The audio file is empty or could not be decoded.')
    if len(y)>sr*30: y=y[:sr*30]
    mfcc=librosa.feature.mfcc(y=y,sr=rate,n_mfcc=20)
    delta=librosa.feature.delta(mfcc);delta2=librosa.feature.delta(mfcc,order=2)
    feats=np.concatenate([mfcc.mean(axis=1),mfcc.std(axis=1),delta.mean(axis=1),delta2.mean(axis=1),[librosa.feature.spectral_centroid(y=y,sr=rate).mean(),librosa.feature.spectral_bandwidth(y=y,sr=rate).mean(),librosa.feature.zero_crossing_rate(y).mean()]])
    return feats.astype(np.float32)

def _load():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH): raise FileNotFoundError("Audio model not found. Put the Audio Sentiment Analysis dataset in data/, then run 'python model/train_audio_model.py'.")
        _model=joblib.load(MODEL_PATH)
    return _model

def analyze_audio(path):
    model=_load();x=_features(path).reshape(1,-1);proba=model.predict_proba(x)[0];classes=list(model.classes_);idx=int(np.argmax(proba));return {'emotion':str(classes[idx]).title(),'confidence':round(float(proba[idx])*100,1),'probabilities':{str(c).title():round(float(p)*100,1) for c,p in zip(classes,proba)}}
