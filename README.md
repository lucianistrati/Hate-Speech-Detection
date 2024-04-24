# Hate Speech Detection

This repository hosts a project titled "Hate Speech Detection," aimed at identifying instances of hate speech in user-generated content, particularly tweets. Hate speech, a prevalent issue in online discourse, targets individuals or groups based on characteristics such as race, religion, ethnicity, sexual orientation, disability, or gender, often serving to degrade or dehumanize them.

## Competition Link

For those interested in the competition associated with this project, further details can be found at: [PAN 2021 Author Profiling Competition](https://pan.webis.de/clef21/pan21-web/author-profiling.html)

## Observations about the Data

The dataset utilized in this project is sourced from the PAN 2021 Author Profiling competition. It consists of profiles from 400 authors, comprising 200 English authors and 200 Spanish authors, each associated with 100 tweets. The authors' profiles are labeled with a binary classification (1 or 0) indicating whether they engage in spreading hate speech. Notably, the dataset is balanced, with 100 authors identified as spreading hate speech and 100 as not engaging in hate speech for both English and Spanish languages.

## Files

- `BERT.py`: Implementation of BERT-based models for hate speech detection.
- `BERT2.py`: Additional BERT-based model variations for hate speech detection.
- `BERT_notebook.ipynb`: Jupyter notebook containing BERT model implementation and experimentation.
- `BERT_notebook_FineTuned.ipynb`: Jupyter notebook focusing on fine-tuning BERT models for hate speech detection.
- `HSDataset.py`: Custom dataset class for hate speech detection.
- `HateSpeechBERT.py`: Module encapsulating BERT-based hate speech detection functionalities.
- `build_emoticon_dict.py`: Script for building an emoticon dictionary.
- `emoji_preprocessor.py`: Module for preprocessing emojis in textual data.
- `en.csv`: Dataset file containing English tweets.
- `fasttext.py`: Implementation of fastText models for hate speech detection.
- `fasttext2.py`: Additional fastText model variations for hate speech detection.
- `fasttext_splitter.py`: Script for splitting fastText models.
- `huge_hs_dataset.csv`: Dataset file containing a large collection of tweets.
- `main.py`: Main script orchestrating the hate speech detection pipeline.
- `read_data.py`: Script for reading dataset files.
- `test.txt`: Test data file.
- `traducer.py`: Module for translation functionalities.
- `train.txt`: Training data file.
- `tweet_preprocessor.py`: Module for preprocessing tweets.
- `Poster.pdf`: Poster presentation summarizing the project.
- `Poster.png`: Image file of the project poster.

Feel free to explore and utilize these files for hate speech detection tasks and related experimentation.

# Project Poster
![Project poster](Poster.png "Project poster")

[![Stand With Ukraine](https://raw.githubusercontent.com/vshymanskyy/StandWithUkraine/main/banner2-direct.svg)](https://stand-with-ukraine.pp.ua)
