from collections import defaultdict, Counter
import csv

def build_features(folder, filename):
    word_id = {"<UNKNOWN>":0}
    id_word = {0:"<UNKNOWN>"}
    tag_id = {}
    id_tag = {}
    for line in open(folder+"/"+filename):
        if line.strip() != "":
            word, tag = line.strip().split("\t")
            if word not in word_id:
                id_word[len(word_id)] = word
                word_id[word] = len(word_id)
            if tag not in tag_id:
                id_tag[len(tag_id)] = tag
                tag_id[tag] = len(tag_id)
    return word_id, id_word, tag_id, id_tag

def vectorize(folder, filename, word_id, tag_id):
    vectors = list()
    vector = [list(),list()]
    for line in open(folder+"/"+filename):
        if line.strip() == "":
            vectors.append(vector)
            vector = [list(),list()]
        else:
            word, tag = line.strip().split("\t")
            if word not in word_id:
                word = "<UNKNOWN>"
            vector[0].append(word_id[word])
            vector[1].append(tag_id[tag])
    vectors.append(vector)
    return vectors

def build_features_lower(folder, filename):
    word_id = {"<UNKNOWN>":0}
    id_word = {0:"<UNKNOWN>"}
    tag_id = {}
    id_tag = {}
    for line in open(folder+"/"+filename):
        if line.strip() != "":
            word, tag = line.strip().split("\t")
            word = word.lower()
            if word not in word_id:
                id_word[len(word_id)] = word
                word_id[word] = len(word_id)
            if tag not in tag_id:
                id_tag[len(tag_id)] = tag
                tag_id[tag] = len(tag_id)
    return word_id, id_word, tag_id, id_tag

def vectorize_lower(folder, filename, word_id, tag_id):
    vectors = list()
    vector = [list(),list()]
    for line in open(folder+"/"+filename):
        if line.strip() == "":
            vectors.append(vector)
            vector = [list(),list()]
        else:
            word, tag = line.strip().split("\t")
            word = word.lower()
            if word not in word_id:
                word = "<UNKNOWN>"
            vector[0].append(word_id[word])
            vector[1].append(tag_id[tag])
    if len(vector[0]) != 0:
        vectors.append(vector)
    return vectors
