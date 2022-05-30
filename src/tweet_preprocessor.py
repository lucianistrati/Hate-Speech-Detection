from emoji_preprocessor import replace_emoji_to_words
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
stop_words = set(stopwords.words('english'))
from string import punctuation
from spellchecker import SpellChecker
spell = SpellChecker(language='en')

# import bert_tokenizer
# from transformers import BertTokenizer

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

import re

def preprocess_tweet(tweet, preprocessing_method, max_seq_len=512):
    tweet = replace_emoji_to_words(tweet)
    if preprocessing_method == 'BERT':
        words_list = word_tokenize(tweet)
        for i in range(len(words_list)):
            if len(spell.unknown([words_list[i]])) != 0:
                words_list[i] = spell.correction(words_list[i])
        return words_list[:max_seq_len] # encoded_sent
    elif preprocessing_method == 'classical':
        # spelling mistakes
        words_list = word_tokenize(tweet)
        for i in range(len(words_list)):
            if len(spell.unknown([words_list[i]])) != 0:
                words_list[i] = spell.correction(words_list[i])

        # cutting punctuation
        tweet = " ".join(words_list)
        no_punctuation_tweet = ""
        for i in range(len(tweet)):
            if tweet[i] not in punctuation:
                no_punctuation_tweet += tweet[i]
        tweet = no_punctuation_tweet

        # tokenizing
        words_list = word_tokenize(tweet)

        # stopwords removal
        words_list = [word for word in words_list if word not in stop_words]

        # stemming
        porter_stemmer = PorterStemmer()
        for i in range(len(words_list)):
            words_list[i] = porter_stemmer.stem(words_list[i])

        return words_list


