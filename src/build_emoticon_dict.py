import pandas as pd
import string

def keep_letters(text):
    lowercase_letters = string.ascii_lowercase
    uppercase_letters = string.ascii_uppercase
    cleaned_text = ""
    for i in range(len(text)):
        if text[i] in lowercase_letters or text[i] ==" " or text[i] in uppercase_letters:
            cleaned_text+=text[i]
    return cleaned_text

emoticon_dataframes = pd.read_html("https://en.wikipedia.org/wiki/List_of_emoticons")

emoticon_dict = {"Emoticon":[], "Meaning":[]}

for i in range(len(emoticon_dataframes)):
    if "Icon" in emoticon_dataframes[i].columns:
        for j in range(len(emoticon_dataframes[i].columns)):
            if emoticon_dataframes[i].columns[j] != "Meaning":
                col = emoticon_dataframes[i].columns[j]
                for k in range(len(emoticon_dataframes[i])):
                    emoticon_dict["Emoticon"].append(emoticon_dataframes[i].iloc[k][col])
                    emoticon_dict["Meaning"].append(keep_letters(emoticon_dataframes[i].iloc[k]['Meaning']))

print(emoticon_dict['Emoticon'][:5], len(emoticon_dict['Emoticon']))
print(emoticon_dict['Meaning'][:5], len(emoticon_dict['Meaning']))

final_dataframe = pd.DataFrame(data=emoticon_dict)
final_dataframe.to_csv("Emoticons_df.csv")
