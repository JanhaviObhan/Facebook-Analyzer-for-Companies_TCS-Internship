import pandas as pd
import numpy as np
import re
import string
import spacy
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim_models
import nltk
from nltk.corpus import stopwords
from gensim.models.coherencemodel import CoherenceModel
from imp import reload

# removes all the special characters (Symbols, emojis) present in the Caption. #
def clean_text(text):
    delete_dict = {sp_character: '' for sp_character in string.punctuation}
    delete_dict[' '] = ' '
    table = str.maketrans(delete_dict)
    text1 = text.translate(table)
    textArr = text1.split()
    text2 = ' '.join([w for w in textArr if (not w.isdigit() and (not w.isdigit() and len(w)>3))])
    
    return text2.lower()

stop_words = stopwords.words('english')

# remove stop words for eg. I, was, an, etc. #
def remove_stopwords(text):
    textArr = text.split(' ')
    rem_text = " ".join([i for i in textArr if i not in stop_words])
    return rem_text


nlp = spacy.load('en_core_web_md', disable=['parser', 'ner'])

# Lematizing words. For eg, Develop, developed, development .. all will get converted to 'Develop' #
def lematization(texts, allowed_postags=['NOUN' , 'ADJ']):
    output = []
    for sent in texts:
        doc = nlp(sent)
        output.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return output

# Coherence Value - Finding the number of topics appropriate for the particular dataset in which the feactures/topics can be fitted properly. 
# A set of statements or facts is said to be coherent, if they support each other. Thus, a coherent fact set can be interpreted in a context that covers all or most of the facts.
# C_v measure is based on a sliding window, one-set segmentation of the top words and an indirect confirmation measure that uses normalized pointwise mutual information (NPMI) and the cosine similarity
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics  in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence = 'u_mass')
        coherence_values.append(coherencemodel.get_coherence())
        
    return model_list, coherence_values




def lda_graph(df):
    df.dropna(axis=0, how='any', inplace=True)
    df['Captions'] = df['Captions'].apply(clean_text)
    df['num_words_text'] = df['Captions'].apply(lambda x:len(str(x).split()))
    df['Captions'] = df['Captions'].apply(remove_stopwords)
    text_list = df['Captions'].tolist()
    tokenized_Captions = lematization(text_list)
    dictionary = corpora.Dictionary(tokenized_Captions)
    doc_term_matrix = [dictionary.doc2bow(rev) for rev in tokenized_Captions]

    model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=doc_term_matrix, texts=tokenized_Captions, start=2, limit=50, step=1)
    max_value = max(coherence_values)
    max_index = coherence_values.index(max_value) + 2

    best_model = model_list[max_index]

    ldamodel= best_model

    vis = pyLDAvis.gensim_models.prepare(best_model, doc_term_matrix, dictionary, mds='mmds')
    filename = "./templates/topics.html"
    pyLDAvis.save_html(vis, filename)
