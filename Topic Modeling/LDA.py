import re
import numpy as np
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from gensim.models.ldamodel import LdaModel
from nltk.corpus import stopwords

data = pd.read_csv('LDA.csv')
documents = data['text']
def preprocess_data(documents):
 stop_words = stopwords.words('english')
 
 # Tokenize and remove stopwords
 texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in documents]
 
 return texts

processed_texts = preprocess_data(documents)

id2word = corpora.Dictionary(processed_texts)
texts = processed_texts
corpus = [id2word.doc2bow(text) for text in texts]

num_topics = 3
lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=42, passes=10, alpha='auto', per_word_topics=True)

pprint(lda_model.print_topics())

coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_texts, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('Coherence Score: ', coherence_lda)