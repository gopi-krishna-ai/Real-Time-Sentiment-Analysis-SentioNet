"""Prepare the Kaggle Sentiment Analysis Dataset placed in data/."""
import os, re, glob
import pandas as pd

BASE=os.path.dirname(os.path.abspath(__file__))
URL_RE=re.compile(r'https?://\S+|www\.\S+',re.I)
MENTION_RE=re.compile(r'@\w+')
NON_ALPHA_RE=re.compile(r"[^a-z\s']")
MULTI_SPACE_RE=re.compile(r'\s+')

def clean_text(x):
    x=str(x).lower();x=URL_RE.sub(' ',x);x=MENTION_RE.sub(' ',x);x=NON_ALPHA_RE.sub(' ',x);return MULTI_SPACE_RE.sub(' ',x).strip()

def find_csv():
    candidates=[p for p in glob.glob(os.path.join(BASE,'*.csv')) if not os.path.basename(p).startswith('sentiment_dataset')]
    if not candidates: raise FileNotFoundError('No training CSV found in data/. Put the Kaggle Sentiment Analysis Dataset CSV in the data folder.')
    return candidates[0]

def pick_columns(df):
    lower={c.lower().strip():c for c in df.columns}
    text=None; label=None
    for c in ['text','tweet','review','sentence','comment','content']:
        if c in lower:text=lower[c];break
    for c in ['sentiment','label','target','polarity','class']:
        if c in lower:label=lower[c];break
    if text is None or label is None: raise ValueError(f'Could not identify text/label columns. Found: {list(df.columns)}')
    return text,label

def normalize_label(v):
    s=str(v).strip().lower()
    mapping={'0':'negative','1':'positive','2':'neutral','negative':'negative','neg':'negative','positive':'positive','pos':'positive','neutral':'neutral'}
    return mapping.get(s,s)

def main():
    path=find_csv();df=pd.read_csv(path,encoding='utf-8',on_bad_lines='skip');text_col,label_col=pick_columns(df)
    out=pd.DataFrame({'text':df[text_col].astype(str).map(clean_text),'label':df[label_col].map(normalize_label)})
    out=out[(out.text.str.split().str.len()>=2)&out.label.notna()].drop_duplicates('text')
    valid=out[out.label.isin(['positive','negative','neutral'])]
    if valid['label'].nunique()<2: raise ValueError('The dataset must contain at least two usable sentiment classes.')
    dest=os.path.join(BASE,'sentiment_dataset.csv');valid.to_csv(dest,index=False);print(f'Source: {path}\nRows: {len(valid)}\nClasses: {valid.label.value_counts().to_dict()}\nSaved: {dest}')
if __name__=='__main__':main()
