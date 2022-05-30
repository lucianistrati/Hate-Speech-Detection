import csv
import glob
import xml.etree.ElementTree as ET
import re
from spellchecker import SpellChecker
from translate import Translator
import os


BI_FOLDER = "pan21-author-profiling-training-2021-03-14"
EN_FOLDER = "en"
ES_FOLDER = "es"


class TweetTranslator:
    def __init__(self, src='en', dst='es'):
        self._translator = Translator(to_lang=dst, from_lang=src)
        self.spell_checker = SpellChecker(language=src)

    def translate_tweet(self, tweet):
        translated_tweet = ""
        splitted_tweet = tweet.replace(',', ' ').split()
        batch = ''
        current_size = 0
        for word in splitted_tweet:
            if current_size + len(word) + 1 < 500:
                batch = batch + word + ' '
                current_size = current_size + 1 + len(word)
            else:
                corrected_batch = self.spell_checker.correction(batch)
                translated_batch = self._translator.translate(corrected_batch)
                translated_tweet = translated_tweet + translated_batch + ' '
                batch = word
                current_size = len(word)
        return translated_tweet


def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


def make_csv_and_translate(path, from_lang, to_lang):
    fields = ['Id', 'Text', 'Label']
    files = glob.glob(path + '\\*.xml')
    truth_path = path + '\\truth.txt'
    label_file = open(truth_path)
    label_dict = dict()
    for k in label_file:
        padding = 1 if k[-1] == '\n' else 0

        current_label = k[len(k) - 1 - padding]
        current_user = k[:(len(k) - 4 - padding)]

        label_dict[current_user] = current_label

    filename = from_lang + '.csv'
    translated_filename = from_lang + '_translated' + '.csv'
    translator = TweetTranslator(from_lang, to_lang)

    csv_file = open(filename, 'w', encoding='utf-8')
    csv_translated_file = open(translated_filename, 'w', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(fields)
    csv_translated_writer = csv.writer(csv_translated_file)
    csv_translated_writer.writerow(fields)
    cnt = 0
    for file in files:
        name = file[len(path) + 1: (len(file) - 4)]
        myTree = ET.parse(file)
        myRoot = myTree.getroot()
        tweet = ""
        for x in myRoot[0]:
            tweet += x.text + ' '
        tweet = remove_emoji(tweet)
        translated_tweet = translator.translate_tweet(tweet)
        label = label_dict[name]
        row = [name, tweet, label]
        translated_row = [name, translated_tweet, label]
        csv_writer.writerow(row)
        csv_translated_writer.writerow(translated_row)
        cnt += 1
    print(cnt)


def double_data():
    en_path = os.path.join(BI_FOLDER, EN_FOLDER)
    es_path = os.path.join(BI_FOLDER, ES_FOLDER)

    make_csv_and_translate(en_path, 'en', 'es')
    make_csv_and_translate(es_path, 'es', 'en')
