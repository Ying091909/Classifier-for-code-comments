import csv
import re
import json
import random
import logging

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import RobertaTokenizer
from collections import Counter
from collections import defaultdict
import os.path
import pandas as pd

nltk.download('wordnet')


# Remove punctuation, extra spaces, special characters, etc.
def remove_punctuation_and_transform_to_prototype(sentence):
    punctuations = '#!?"$%&()*,./;:^<=>[\\]+-'
    replace_str = " " * len(punctuations)
    a = ' '.join(sentence.translate(str.maketrans(punctuations, replace_str)).split())
    return a


def lemmatize_all_data():
    # stop = open("./stop_word.txt").readlines()
    for j in ['java', 'pharo', 'python']:
        df = pd.read_csv(j + ".csv")
        write_file = "dataset/pre" + "_" + j + ".csv"
        with open(write_file, 'w', newline="") as file:
            write = csv.writer(file)
            header = df.columns.values.tolist()
            header.append("pre_sentence")
            write.writerow(header)
            for i in df.values.tolist():
                row = i
                line = remove_punctuation_and_transform_to_prototype(i[2])
                row.append(line)
                write.writerow(row)


def split_train_and_val_dataset(lan, c, new_data):
    # # 划分训练集和验证集时，对预处理后的comment进行去重
    # seen = set()
    # new_data = []
    # for row in data_of_one:
    #     if row[6] not in seen:
    #         seen.add(row[6])
    #         new_data.append(row)
    train_info, val_info = dict(), dict()
    train_data, val_data = [], []
    new_data.sort(key=lambda x: x[4], reverse=True)
    true_count = sum([i[4] for i in new_data])
    false_count = len(new_data) - true_count
    ratio = true_count / len(new_data)
    val_data = random.sample(new_data[:true_count + 1], int(true_count * ratio)) + \
               random.sample(new_data[true_count + 1:], int(false_count * ratio))
    for i in new_data:
        if i not in val_data:
            train_data.append(i)
    true_of_train = 0
    true_of_val = 0
    for v in val_data:
        if v[4] == 1:
            true_of_val += 1
    for t in train_data:
        if t[4] == 1:
            true_of_train += 1
    train_info['positive'] = true_of_train
    train_info['negative'] = len(train_data) - true_of_train
    train_info['total'] = len(train_data)
    val_info['positive'] = true_of_val
    val_info['negative'] = len(val_data) - true_of_val
    val_info['total'] = len(val_data)
    print("===============================")
    print("%s %s " % (lan, c))
    print("all data", len(new_data))
    print("train_data:", train_info)
    print("val_data:", val_info)
    print("all true count", true_count)
    return val_data, train_data


def split_data_from_csv(language, filename, categories):
    data = pd.read_csv("dataset/" + filename).values.tolist()
    all_train_data = dict.fromkeys(categories)
    test_data = dict.fromkeys(categories)
    for k in test_data.keys():
        all_train_data[k] = []
        test_data[k] = []
    for i in data:
        category = i[5]
        if i[3] == 0:
            all_train_data[category].append(i)
        else:
            test_data[category].append(i)
    for j in range(len(categories)):
        random.shuffle(all_train_data[categories[j]])
        random.shuffle(test_data[categories[j]])
    header = ['comment_sentence_id', 'class', 'comment_sentence', 'partition', 'instance_type', 'category',
              'pre_sentence']
    csv_files_path = dict.fromkeys(categories)
    for c in categories:
        val_data, train_data = split_train_and_val_dataset(language, c, all_train_data[c])
        train_file = "./dataset/" + language + "/train_data_of_" + c + ".csv"
        val_file = "./dataset/" + language + "/val_data_of_" + c + ".csv"
        test_file = "./dataset/" + language + "/test_data_of_" + c + ".csv"
        with open(train_file, 'w', newline='') as f1:
            writer = csv.writer(f1)
            writer.writerow(header)
            writer.writerows(train_data)
        with open(val_file, 'w', newline='') as f2:
            writer = csv.writer(f2)
            writer.writerow(header)
            writer.writerows(val_data)
        with open(test_file, 'w', newline='') as f3:
            writer = csv.writer(f3)
            writer.writerow(header)
            writer.writerows(test_data[c])
        csv_files_path[c] = [train_file, val_file, test_file]
    return csv_files_path


def reformat_data(filename, target_file, lan, c, mode):
    """
    transform the data format to jsonl
    """
    data = pd.read_csv(filename).values.tolist()
    if os.path.exists(target_file):
        os.remove(target_file)
    for i in data:
        with open(target_file, 'a+') as file:
            content = {}
            # comment_sentence_id	class	comment_sentence	partition	instance_type	category  pre_sentence
            content["comment_sentence_id"] = i[0]
            content["class"] = i[1]
            content["comment_sentence"] = i[2]
            content["partition"] = i[3]
            content["instance_type"] = i[4]
            content["category"] = i[5]
            content["pre_sentence"] = i[6]
            # mark_sentence = ""
            # # features = open(lan + "_" + c + "_" + mode + "_arff.txt").readlines()
            # # features = open("./arrf/" + lan + "_" + mode + ".arff").readlines()
            mark_sentence = str(i[6])
            features = get_pattern("./arrf/" + lan + "_" + mode + ".arff")[c]
            # # 对存在arff中的词进行标记<>
            for f in features:
                if len(f) < 3:
                    break
                if f in str(i[6]):
                    mark_sentence = str(i[6]).replace(f, "<s>" + f + "</s>")
            content["final_sentence"] = mark_sentence
            json.dump(content, file)
            file.write("\n")


# Get the Expert-predefined features from the arff file
def get_pattern(path):
    data = open(path).readlines()
    patterns = defaultdict(list)
    for i in data:
        if "{0,1}" in i and i.startswith("@attribute"):
            line = i.replace("{0,1}\n", "").split("-")
            if len(line) > 2:
                pattern = remove_punctuation_and_transform_to_prototype(re.sub(r'\[.*?\]', '', line[2].lower()))
                if len(pattern) > 0:
                    patterns[line[1]].append(pattern.replace(" '", ""))
    return patterns


# Write the feature for each category to a txt file
def write_pattern_into_file():
    for _, _, files in os.walk("./arrf"):
        for f in files:
            info = f.replace(".arff", "").split("_")
            language, data_type = info[0], info[1]
            patterns = get_pattern("./arrf/" + f)
            for c, p in patterns.items():
                write_file = open(language + "_" + c + "_" + data_type + "_arff.txt", 'w')
                write_file.writelines(patterns[c])


if __name__ == "__main__":
    lemmatize_all_data()
    # write_pattern_into_file()
    multi_class = {'java': ['Ownership', 'Expand', 'usage', 'Pointer', 'deprecation', 'rational', 'summary'],
                   'pharo': ['Keymessages', 'Intent', 'Classreferences', 'Example', 'Keyimplementationpoints',
                             'Responsibilities', 'Collaborators'],
                   'python': ['Summary', 'Parameters', 'Usage', 'DevelopmentNotes', 'Expand']}
    for lan in multi_class.keys():
        print("********************************************")
        print("start preprocessing the data of %s" % lan)
        # split each class of data into train data and test data // train val test
        csv_files_path = split_data_from_csv(lan, "pre_" + lan + '.csv', multi_class[lan])
        # transform data format from csv to jsonl
        print("all categories files:", csv_files_path)
        for index, (c, v) in enumerate(csv_files_path.items()):
            mode = 'test' if index == 2 else "train"
            print("category:", c)
            for j in v:
                reformat_data(j, j.replace(".csv", ".jsonl"), lan, c, mode)
        print("preprocessing about %s has fininshed !" % lan)




