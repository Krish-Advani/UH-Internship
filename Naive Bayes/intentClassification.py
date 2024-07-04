import pandas as pd
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report



#Cleaning Data
data = pd.read_csv('intent.csv')
data['question']=data['question'].str.lower()
data['class']=data['class'].str.lower()
lemmatizer = WordNetLemmatizer()
data['lemQuestion'] = data['question'].apply(lambda sentence:[lemmatizer.lemmatize(token) for token in sentence.split()])
def removePunc(words_list):
    pattern = r'[^a-zA-Z0-9\s]'
    clean_words=[re.sub(pattern,"",word) for word in words_list]
    return clean_words
data['cleanQuestion']=data['lemQuestion'].apply(removePunc)
data['joinQuestion']=data['cleanQuestion'].apply(lambda x: ' '.join(x))



#Train Model
X_train,X_test,y_train,y_test=train_test_split(data['joinQuestion'],data['class'],test_size=0.2,random_state=42)
vectorizer = TfidfVectorizer()
X_train_vectorized=vectorizer.fit_transform(X_train)
X_test_vectorized=vectorizer.transform(X_test)
nb=MultinomialNB()
nb.fit(X_train_vectorized,y_train)



#Evaluate Model Accuracy
y_pred = nb.predict(X_test_vectorized)
accuracy = accuracy_score(y_test,y_pred)
report = classification_report(y_test,y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:\n",report)