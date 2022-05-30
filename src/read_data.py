from xml.dom import minidom
from random import shuffle
import os


def read_tweets_of_author(file_path, num_tweets_per_batch=100):
    """
    :param file_path: str, the path of the xml file that contains the tweets of
                        of an author
    :return: author_list: list of str, where each element of the list is a tweet
                            of an author
             tweets_polarity: int, 1 - if the author has any tweets that incite
                            to hate speech, 0 otherwise
    """
    if file_path.endswith(".xml") == False:
        return None, None
    doc = minidom.parse(file_path)
    items = doc.getElementsByTagName("document")
    single_author_tweets_list = []
    tweets_polarity = doc.getElementsByTagName("author")[0].getAttribute("class")

    tweets_polarity = int(tweets_polarity)

    cnt = 0
    debugging = False
    MAX_CHECK = 10

    for item in items:
        if '(' in item.firstChild.data:
            debugging = True
        cnt += 1
        if cnt > MAX_CHECK:
            break

    debugging = False
    cnt = 0
    for item in items:
        single_author_tweets_list.append(item.firstChild.data)
        if debugging:
            print('Step #', cnt, ':')
            print('First Child:', item.firstChild.data)
            print('List:', single_author_tweets_list)
            cnt += 1
            if cnt > MAX_CHECK:
                exit(-1)
    if num_tweets_per_batch == 100:
        return [". ".join(single_author_tweets_list)], tweets_polarity
    elif num_tweets_per_batch == 1:
        return single_author_tweets_list, tweets_polarity
    elif 2 <= num_tweets_per_batch <= 99:
        shuffle(single_author_tweets_list)
        batch_tweets_list = []
        for i in range(100 // num_tweets_per_batch):
            batch_tweets_list.append(". ".join(single_author_tweets_list[i: 100: 100 // num_tweets_per_batch]))
        return batch_tweets_list, tweets_polarity
    else:
        print("An invalid number of tweets was given!")
        return None, None


def read_folder_of_authors(folder_path, args, dataset_type=''):
    """
    :param folder_path: str, the path of the folder that contains the files of
                            multiple authors
    :return: authors_list: list of str, where each element of the list
                            is a list with a batch of tweets posted by an author,
                            batch that can consist of 1 tweet or 100 tweets
             tweets_labels_list: list of int, where each element is either 1
                            or 0  if the author at the same position in the
                            authors_list incites to hate speech or not
    """
    authors_list = []
    tweets_labels_list = []
    for (dirpath, dirnames, filenames) in os.walk(folder_path):
        files = sorted(filenames)
        if dataset_type == 'train':
            left_idx = 0
            right_idx = int(args.train_data_pct * len(files))
        elif dataset_type == 'val':
            left_idx = int(args.train_data_pct * len(files))
            right_idx = int((args.train_data_pct + args.val_data_pct) * len(files))
        elif dataset_type == 'test':
            left_idx = int((args.train_data_pct + args.val_data_pct) * len(files))
            right_idx = len(files)
        else:
            left_idx = 0
            right_idx = len(files)
        author_counter = 0
        for i in range(left_idx, right_idx):
            filename = files[i]
            author_counter += 1
            single_author_tweets_list, tweets_polarity = read_tweets_of_author(os.path.join(dirpath, filename), args.num_tweets_per_batch)
            if single_author_tweets_list is not None and tweets_polarity is not None:
                for batch_tweets_list in single_author_tweets_list:
                    authors_list.append(batch_tweets_list)
                    tweets_labels_list.append(tweets_polarity)
        print("The tweets of " + str(author_counter) + " authors were read")
    return authors_list, tweets_labels_list

