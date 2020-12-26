import numpy as np
import nltk
import string
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def fix_contraction ( pre_text ):
    pre_text['no_contract'] = pre_text['text'].apply(lambda x: [contractions.fix(word) for word in x.split()])

def join_post_contra ( pre_text ):
    pre_text['text_str'] = [' '.join(map(str, l)) for l in pre_text['no_contract']]

def remove_stopwords ( pre_text ):
    stop_words = set(stopwords.words('english'))
    pre_text['stopwords_removed'] = pre_text['no_punc'].apply(lambda x: [word for word in x if word not in stop_words])

def remove_punc ( pre_text ):
    punc = string.punctuation
    pre_text['no_punc'] = pre_text['lower_str'].apply(lambda x: [word for word in x if word not in punc])

def change_to_lower ( pre_text ):
    pre_text['lower_str'] = pre_text['tokenized'].apply(lambda x: [word.lower() for word in x])

def text_tokenize ( pre_text ):
    #print (pre_text)
    pre_text['tokenized'] = pre_text['text_str'].apply(word_tokenize)

def lemma ( pre_text ):
    pre_text['pos_tags'] = pre_text['stopwords_removed'].apply(nltk.tag.pos_tag)
    pre_text['wordnet_pos'] = pre_text['pos_tags'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])
    wnl = WordNetLemmatizer()
    pre_text['lemmatized'] = pre_text['wordnet_pos'].apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])

def data_processing ( text ):
    p_text = text.copy(deep=False)
    fix_contraction ( p_text )
    join_post_contra ( p_text )
    text_tokenize ( p_text )
    change_to_lower ( p_text )
    remove_punc ( p_text )
    remove_stopwords ( p_text )
    lemma ( p_text )
    p_text['final_words'] = [' '.join(map(str, l)) for l in p_text['lemmatized']]
    pr_text = p_text['final_words']
    return pr_text

