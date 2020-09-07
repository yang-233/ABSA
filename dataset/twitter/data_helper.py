# -*- coding: utf-8 -*-
"""
  File Name : data_helper 
  Author : Hemeng
  date : 2020/2/21
  Description :
  Change Activity: 2020/2/21

"""
from ABS.config import *
from ABS.generate_wordlist import *
from sklearn import metrics

import numpy as np
import os
import pickle
import random


def load_data(domain, train_string, shuffle=True):
    """
    :param train_string:  train or all
    :return:
    """
    if not os.path.exists(config.save_path + "dataset/"):
        os.makedirs(config.save_path + "dataset/")
    filename = config.save_path + "dataset/" + domain + "_" + train_string
    if not os.path.exists(filename):
        if train_string == "train":
            reviews, reviews_len, labels, aspects = process_data(domain, train_flag=True, shuffle=shuffle)
        elif train_string == "test":
            reviews, reviews_len, labels, aspects = process_data(domain, train_flag=False, shuffle=shuffle)
        else:
            reviews, reviews_len, labels, aspects = process_data(domain, train_flag=True, shuffle=shuffle)
            reviews_, reviews_len_, labels_, aspects_ = process_data(domain, train_flag=False,
                                                                     shuffle=shuffle)
            # reviews = np.concatenate([reviews, reviews_], axis=0)
            # reviews_len = np.concatenate([reviews_len, reviews_len_], axis=0)
            # labels = np.concatenate([labels, labels_], axis=0)
            # aspects = np.concatenate([aspects, aspects_], axis=0)
            # position = np.concatenate([position, position_], axis=0)
            reviews.extend(reviews_)
            reviews_len.extend(reviews_len_)
            labels.extend(labels_)
            aspects.extend(aspects_)

        write_f = open(filename, 'wb')
        pickle.dump(np.asarray(reviews), write_f, -1)
        pickle.dump(np.asarray(reviews_len), write_f)
        pickle.dump(np.asarray(labels), write_f)
        pickle.dump(np.asarray(aspects), write_f)

        print("Finished to write to file :%s" % (filename))
        write_f.close()

    read_f = open(filename, 'rb')
    reviews = pickle.load(read_f)
    reviews_len = pickle.load(read_f)
    labels = pickle.load(read_f)
    aspects = pickle.load(read_f)

    return reviews, reviews_len, labels, aspects


def process_data(domain, train_flag, shuffle=True):
    if train_flag:
        dataset = config.ABS_dataset + domain + "_Train.raw"
    else:
        dataset = config.ABS_dataset + domain + "_Test.raw"
    reviews = []
    labels = []
    position = []
    aspect_text = []

    with open(dataset, encoding="utf-8", mode="r") as f:
        for line_i, line in enumerate(f):

            if line_i % 3 == 0:

                text = line
                text = cleanSentences(text)

            elif line_i % 3 == 1:
                aspect = line.replace("\n", "").strip()
                aspect = cleanSentences(aspect)
                aspects = aspect.split()
                texts = text.split()
                cur_pos = [-1] * (len(texts) + len(aspects) - 1)
                for i in range(len(texts)):
                    if texts[i] == "$t$":
                        cur_pos[i] = 1
                        for j in range(len(aspects)):
                            cur_pos[i + j] = 0
                        break
                text = text.replace("$t$", aspect)
            elif line_i % 3 == 2:
                cur_label = int(line.strip())
                reviews.append(text)
                labels.append(cur_label)
                position.append(cur_pos)
                aspect_text.append(aspect)

    f.close()
    word2id = load_word2id()
    reviews, reviews_len = process_reviews(reviews, word2id)
    aspect_text, _ = process_aspects(aspect_text, word2id)

    labels = process_label(labels)

    print("reviews:%d\t reviews_len:%d\t sentiment_labels:%d\t aspect_text:%d\t aspect_labels:%d" % (
        len(reviews), len(reviews_len), len(labels), len(aspect_text), len(position)))
    # if shuffle:
    #     reviews, reviews_len, labels,aspect_text, position = shuffle_data(reviews, reviews_len, labels,aspect_text, position)

    # return np.asarray(reviews), np.asarray(reviews_len), np.asarray(labels), np.asarray(aspect_text),np.asarray(position)
    return reviews, reviews_len, labels, aspect_text  # , position


def process_label(labels):
    """
    labels:-1,0,1
    process each label as [1,0,0],[0,1,0],[0,0,1]
    :return:
    """

    y = []
    for line_i, line in enumerate(labels):
        as_y = [0., 0., 0.]  # each word
        as_y[line + 1] = 1.0
        y.append(as_y)
    return y


def process_aspects(file, word2id):
    aspects = []
    aspects_len = []
    for line in file:
        text = np.zeros(config.max_asp_len)
        words = line.replace("\n", "").strip().split()
        for word_index, word in enumerate(words):
            if word_index < config.max_asp_len:
                if word in word2id:
                    text[word_index] = word2id[word]
                else:
                    print("not in word2id")  # text[word_index] = len(word2id)
            else:
                break
        for i in range(word_index + 1, config.max_asp_len):
            text[i] = len(word2id)
        aspects.append(text)
        aspects_len.append(len(words))
    print("The length of reviews: %d" % (len(aspects)))
    return aspects, aspects_len


def process_reviews(file, word2id):
    """
    process the raw reviews as id

    :return:
    """
    reviews = []
    reviews_len = []
    for line in file:
        text = np.zeros(config.max_seq)  # value is id
        words = line.replace("\n", "").strip().split()
        if len(words) < 2:
            print("The num of words in this line less than 2: " + line)
            continue
        for word_index, word in enumerate(words):
            if word_index < config.max_seq:
                if word in word2id:
                    text[word_index] = word2id[word]
                else:
                    print("not in word2id")  # text[word_index] = len(word2id)
            else:
                break
        for i in range(word_index + 1, config.max_seq):
            text[i] = len(word2id)
        reviews.append(text)
        reviews_len.append(len(words))
    print("The length of reviews: %d" % (len(reviews)))
    return reviews, reviews_len


def shuffle_data(x1, x2, x3, x4):
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    x3 = np.asarray(x3)
    x4 = np.asarray(x4)

    size = len(x1)
    shuffle_idx = np.random.permutation(np.arange(size))
    shuffle_x1 = x1[shuffle_idx]
    shuffle_x2 = x2[shuffle_idx]
    shuffle_x3 = x3[shuffle_idx]
    shuffle_x4 = x4[shuffle_idx]

    return shuffle_x1, shuffle_x2, shuffle_x3, shuffle_x4


def enlarge_data(x1, x2, x3, x4, increase_num):
    x1, x2, x3, x4 = list(x1), list(x2), list(x3), list(x4)
    size = len(x1)
    if increase_num > size:
        temp = increase_num // size
        for _ in range(temp):
            x1.extend(x1)
            x2.extend(x2)
            x3.extend(x3)
            x4.extend(x4)
        increase_choice = random.sample(range(0, size), increase_num - temp * size)
    else:
        increase_choice = random.sample(range(0, size), increase_num)

    for i in increase_choice:
        x1.append(np.asarray(x1[i]))
        x2.append(np.asarray(x2[i]))
        x3.append(np.asarray(x3[i]))
        x4.append(np.asarray(x4[i]))

    print(
        "After enlarged,the length :", len(x1), len(x2), len(x3), len(x4))
    return np.asarray(x1), np.asarray(x2), np.asarray(x3), np.asarray(x4)


def enlarge_all_data(x1, x2, x3, x4, y1, y2, y3, y4):
    data_size = max(len(x1), len(y1))
    if len(x1) < len(y1):
        needed = len(y1) - len(x1)
        x1, x2, x3, x4 = enlarge_data(x1, x2, x3, x4, needed)

    elif len(y1) < len(x1):
        needed = len(x1) - len(y1)
        y1, y2, y3, y4 = enlarge_data(y1, y2, y3, y4, needed)
    return data_size, x1, x2, x3, x4, y1, y2, y3, y4


def batches_iter(x, x_len, y, aspects, batch_size, shuffle=True):
    data_size = len(x)
    batch_num = (data_size // batch_size)
    x = np.array(x)
    x_len = np.array(x_len)
    y = np.array(y)
    aspects = np.array(aspects)
    # asp_pos = np.array(asp_pos)
    if shuffle:
        x, x_len, y, aspects = shuffle_data(x, x_len, y, aspects)  # , asp_pos)
    for batch in range(batch_num):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, data_size)
        yield x[start_idx:end_idx], x_len[start_idx:end_idx], y[start_idx:end_idx], aspects[
                                                                                    start_idx:end_idx]  # , asp_pos[start_idx:end_idx]


def cal_acc_and_f1(s_pred, s_labels):
    return metrics.accuracy_score(s_labels, s_pred), metrics.f1_score(s_labels, s_pred, average='macro')
