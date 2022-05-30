import pandas as pd
import emot
import demoji
import emoji
demoji.download_codes()

def load_emoticon_list_from_df():
    df = pd.read_csv("Emoticons_df.csv")
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    emoticon_list = df.to_dict('records')
    return emoticon_list

global emoticon_dict
emoticon_list = load_emoticon_list_from_df()

def replace_emoji_to_words(tweet):
    global emoticon_list
    # emoji_dict = emoji.demojize(tweet, delimiters=("", ""))
    demoji_dict = demoji.findall(tweet)
    for key, value in demoji_dict.items():
        tweet = tweet.replace(key, value)
    for i in range(len(emoticon_list)):
        if "str" in str(type(emoticon_list[i]['Emoticon'])) and "str" in str(type(emoticon_list[i]['Meaning'])):
            tweet = tweet.replace(emoticon_list[i]['Emoticon'], emoticon_list[i]['Meaning'])
    return tweet

def main():
    text = "text  :-)  :)  :(   =)) 🔥 🔥"
    replace_emoji_to_words(text)

if __name__=="__main__":
    main()