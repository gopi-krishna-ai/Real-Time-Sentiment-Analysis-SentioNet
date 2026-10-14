"""Train an MFCC-based audio emotion classifier from audio files in data/. Labels are inferred from parent folder names."""
import os,glob,joblib,re
import numpy as np,librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));DATA=os.path.join(BASE,'data');OUT=os.path.join(BASE,'model','audio_sentiment_model.pkl')
EXTS={'.wav','.mp3','.m4a','.flac','.ogg','.aac'}

def features(path,sr=22050):
    y,rate=librosa.load(path,sr=sr,mono=True)
    if y.size==0:return None
    y=y[:sr*30]
    mfcc=librosa.feature.mfcc(y=y,sr=rate,n_mfcc=20);delta=librosa.feature.delta(mfcc);delta2=librosa.feature.delta(mfcc,order=2)
    return np.concatenate([mfcc.mean(1),mfcc.std(1),delta.mean(1),delta2.mean(1),[librosa.feature.spectral_centroid(y=y,sr=rate).mean(),librosa.feature.spectral_bandwidth(y=y,sr=rate).mean(),librosa.feature.zero_crossing_rate(y).mean()]]).astype(np.float32)

def infer_label(path):
    parts=[x.lower() for x in os.path.normpath(path).split(os.sep)]
    known=['angry','anger','happy','happiness','sad','sadness','neutral','calm','fear','fearful','disgust','surprise','surprised','excited','excitement','positive','negative']
    for part in reversed(parts):
        for label in known:
            if re.search(rf'(^|[^a-z]){re.escape(label)}([^a-z]|$)',part):
                return label
    return os.path.basename(os.path.dirname(path)).strip()

def main():
    paths=[p for p in glob.glob(os.path.join(DATA,'**','*'),recursive=True) if os.path.splitext(p)[1].lower() in EXTS and os.path.isfile(p)]
    if not paths: raise FileNotFoundError('No audio files found under data/. Extract the Kaggle Audio_Sentiment_Analysis dataset into data/ first.')
    X=[];y=[];skipped=0
    for i,p in enumerate(paths,1):
        label=infer_label(p)
        if not label or label.lower()=='data':
            skipped+=1;continue
        try:
            f=features(p)
            if f is not None:X.append(f);y.append(label)
        except Exception as exc:
            skipped+=1
        if i%250==0:print(f'Processed {i}/{len(paths)} audio files...')
    X=np.vstack(X);y=np.array(y);print(f'Usable samples: {len(y)} | classes: {dict(zip(*np.unique(y,return_counts=True)))}')
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    clf=RandomForestClassifier(n_estimators=350,random_state=42,n_jobs=-1,class_weight='balanced_subsample');clf.fit(Xtr,ytr);pred=clf.predict(Xte)
    print(f'Test accuracy: {accuracy_score(yte,pred):.2%}');print(classification_report(yte,pred,zero_division=0));joblib.dump(clf,OUT);print(f'Saved: {OUT}');print(f'Skipped: {skipped}')
if __name__=='__main__':main()
