import pandas as pd


df = pd.read_csv("en.csv")
print(df.head())

train_file = open("train.txt", "a")
test_file = open("test.txt","a")

for i in range(len(df)):
    label = df.loc[i]['Label']
    if i <= int(0.8 * len(df)):
        train_file.write("__label__" + str(label) + " , " + df.loc[i]['Text']
                         + "\n")
    else:
        test_file.write("__label__" + str(label) + " , " + df.loc[i]['Text']
                         + "\n")

train_file.close()
test_file.close()