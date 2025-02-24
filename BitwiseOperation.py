import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
corpus=['this is the first document.','this document is second document.',
        'and this is the third one.','is this the first document?', ]
vectorizer= CountVectorizer()
X=vectorizer.fit_transform(corpus)
print("fit transform is ")
print(X.toarray())
df=pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names_out())
print("the generated data frame is")
print(df)
alldata= df[(df['this']==1)&(df['first']==1)]
print("indices where 'this'and 'first'terms are present are ",alldata.index.tolist())
ordata= df[(df['this']==1)|(df['first']==1)]
print("indices where either of 'this'and 'first'terms are present are ",ordata.index.tolist())
notdata=df[(df['and']!=1)]
print("indices where 'and' term is not present ",notdata.index.tolist())
