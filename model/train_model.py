"""Train TF-IDF + Logistic Regression on data/sentiment_dataset.csv."""
import os,joblib,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));DATA_PATH=os.path.join(BASE,'data','sentiment_dataset.csv');VEC=os.path.join(BASE,'model','tfidf_vectorizer.pkl');MODEL=os.path.join(BASE,'model','sentiment_model.pkl')
def main():
    df=pd.read_csv(DATA_PATH).dropna(subset=['text','label']);df=df[df.label.astype(str).str.len()>0]
    Xtr,Xte,ytr,yte=train_test_split(df.text.astype(str),df.label.astype(str),test_size=.2,random_state=42,stratify=df.label)
    vec=TfidfVectorizer(max_features=40000,ngram_range=(1,2),min_df=2,sublinear_tf=True);A=vec.fit_transform(Xtr);B=vec.transform(Xte)
    model=LogisticRegression(max_iter=1500,C=1.5);model.fit(A,ytr);pred=model.predict(B)
    print(f'Test accuracy: {accuracy_score(yte,pred):.2%}');print(classification_report(yte,pred,zero_division=0))
    joblib.dump(vec,VEC);joblib.dump(model,MODEL);print('Saved vectorizer and model.')
if __name__=='__main__':main()
