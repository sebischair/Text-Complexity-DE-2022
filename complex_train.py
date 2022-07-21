#!/usr/bin/env python3
# complex_train.py
# train and val on complex DE

import pandas as pd
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
import json
from collections import Counter
import csv
import math
from pathlib import Path
import re
import random

from sklearn import datasets, linear_model, svm
from sklearn.ensemble import RandomForestRegressor, IsolationForest, StackingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel, RationalQuadratic, DotProduct
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV, RepeatedKFold
from sklearn.metrics import mean_squared_error, SCORERS, make_scorer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.decomposition import PCA

import pickle
from pandarallel import pandarallel
pandarallel.initialize()

import spacy
nlp = spacy.load("de_core_news_lg")
STOP = stopwords.words('german')

import wn
wn.download("odenet")
import wn.similarity

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("cardiffnlp/twitter-xlm-roberta-base")

# grid search scorer
mse_pos = make_scorer(mean_squared_error, greater_is_better=False, squared=False)

# preload spaCy tags
pos_tags = "$(, $,, $., ADJA, ADJD, ADV, APPO, APPR, APPRART, APZR, ART, CARD, FM, ITJ, KOKOM, KON, KOUI, KOUS, NE, NN, NNE, PDAT, PDS, PIAT, PIS, PPER, PPOSAT, PPOSS, PRELAT, PRELS, PRF, PROAV, PTKA, PTKANT, PTKNEG, PTKVZ, PTKZU, PWAT, PWAV, PWS, TRUNC, VAFIN, VAIMP, VAINF, VAPP, VMFIN, VMINF, VMPP, VVFIN, VVIMP, VVINF, VVIZU, VVPP, XY".split(', ')
pos_tags = [x.strip() for x in pos_tags]

dep_names = "ROOT, ac, adc, ag, ams, app, avc, cc, cd, cj, cm, cp, cvc, da, dep, dm, ep, ju, mnr, mo, ng, nk, nmc, oa, oc, og, op, par, pd, pg, ph, pm, pnc, punct, rc, re, rs, sb, sbp, svp, uc, vo".split(", ")

# wiki word frequencies
words = {}
with open("result.txt", 'r') as f:
    for l in f.readlines():
        words[l.split()[0]] = int(l.split()[1])

# common German words
common = None
common_map = {}
with open("common_clean.txt", 'r') as f:
    common = [x.strip().lower() for x in f.readlines()]
    c_len = len(common)
    for i, c in enumerate(common):
        common_map[c] = c_len - i

# scrabble values
scrabble = pd.read_csv("scrabble.csv", index_col=[0])
old_scrabble = dict(zip(scrabble.index.str.lower(), scrabble['Face_value_old']))
new_scrabble = dict(zip(scrabble.index.str.lower(), scrabble['Face_value_new']))

# return scrabble score of sentence using new point values
def get_scrabble_score_new(word):
    score = 0
    for w in word:
        if w.lower() in new_scrabble:
            score += new_scrabble[w.lower()]
    return score

# return scrabble score of sentence using old point values
def get_scrabble_score_old(word):
    score = 0
    for w in word:
        if w.lower() in old_scrabble:
            score += old_scrabble[w.lower()]
    return score

# min max normalization
def min_max(vals):
    vals = (vals - vals.min()) / (vals.max() - vals.min()) * 5 + 1
    return vals

# apply log to input values
def log_or_no(val):
    do_it = True
    if do_it == True:
        return np.log(val)
    else:
        return val

# find # of tokens > n characters
def token_chars_gt(sentence, n):
    sum = 0
    for t in sentence.split():
        if len(t) > n:
            sum += 1
    return sum / len(sentence.split())

# cumulative length of dependencies in sentence
def dep_parse_length(sentence):
    doc = nlp(sentence)
    sum = 0
    for d in doc:
        sum += len([x for x in d.subtree])
    return sum

# total length of named entities
def ne_length(sentence):
    doc = nlp(sentence)
    sum = 0
    c = 0
    for e in doc.ents:
        c += 1
        sum += e.end_char - e.start_char
    if c > 0:
        return sum
    else:
        return 0

# total number of named entities
def ne(sentence):
    doc = nlp(sentence)
    return sum([1 for x in doc.ents])

# vector norm sum
def l2norm(sentence):
    doc = nlp(sentence)
    sum = 0
    for d in doc:
        sum += d.vector_norm
    return sum

# does a spaCy vector exist?
def vec_exists(sentence):
    doc = nlp(sentence)
    return sum([1 for x in doc if x.has_vector == True])

# get pos of head word
def get_case(sentence):
    doc = nlp(sentence)
    for d in doc:
        if d.text == d.sent.root.text:
            return d.pos_

# get the syntax trom from nltk
def syn_tree(sentence):
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    entities = nltk.chunk.ne_chunk(tagged)
    return entities

# total distance between the leaves of a sentence
def leaf_distance(sentence):
    tree = syn_tree(sentence)
    n_leaves = len(tree.leaves())
    
    leaf_pos = []
    for l in range(0, n_leaves):
        leaf_pos.append(tree.leaf_treeposition(l)[0])
    
    distance = 0
    for p, idx in enumerate(leaf_pos):
        if idx == 0:
            continue
        
        distance += (leaf_pos[idx] - leaf_pos[idx-1])
    
    return distance

# count the syllables of a word, using the consonants 
def syllable(word):
    syllables = 'aeiouäöü'
    
    count = 0
    for idx in range(0, len(word)):
        if (idx == 0) or (idx != 0 and word[idx-1] not in syllables):
            if word[idx] in syllables:
                count += 1
    
    return count

# linsear metric
def linsear(entry):
    score = entry['easy_words'] + 3*entry['hard_words']
    if score > 20:
        return score / 2
    else:
        return (score / 2) - 1

# number of synsets existing for words in the sentence
def get_synsets(sentence):
    doc = nlp(sentence)
    sets = []
    for d in doc:
        pos = d.pos_[0].lower()
        found = wn.synsets(d.text, pos=pos)
        if len(found) > 0:
            sets.append(found[0])
    return sets

# cumulative maximum depth of synsets        
def get_max_depth(sets):
    if len(sets) == 0:
        return 0
    
    score = 0
    for s in sets:
        score += s.max_depth()
    return score

# total number of hyponyms from synsets
def get_hypo(sets):
    h = 0
    for s in sets:
        h += len(s.hyponyms())
    return h

# total number of senses from synsets
def get_senses(sets):
    ss = 0
    for s in sets:
        ss += len(s.senses())
    return ss

# total length of all synset definitions
def get_def(sets):
    if len(sets) == 0:
        return 0
    length = len(sets)
    tot = 0
    for s in sets:
        if s.definition() is not None:
            tot += len(s.definition())
        else:
            length -= 1
    return tot

# cumulative path length from one synset to another
def path(sets):
    paths = []
    for i, s in enumerate(sets[1:]):
        for j, x in enumerate(sets):
            if i == j:
                continue
            try:
                temp = wn.similarity.wup(s, x, simulate_root=True)
                paths.append(temp)
            except:
                pass
    if len(paths) > 0:
        return sum(paths) / len(sets)
    else:
        return 0

# create feature set for input data
def process(data):
    best_pos = ['ADJA', 'ADJD', 'ADV', 'APPR', 'APPRART', 'ART', 'KON', 'KOUS', 'NN', 'PRELS', 'VAFIN', 'VVFIN', 'VVPP']

    regex = re.compile('[^a-zA-Z]')

    data['Sentence_proc'] = data['Sentence'].apply(lambda x: " ".join([regex.sub('', y) for y in x.split() if regex.sub('', y.lower()) not in STOP]))
    data['Sentence_lemma'] = data['Sentence_proc'].apply(lambda x: " ".join([y.lemma_ for y in nlp(x)]))
    data['sentence_length'] = data['Sentence'].apply(lambda x: log_or_no(len(str(x))))
    
    data['char_token'] = data['Sentence'].apply(lambda x: log_or_no(len(x) / len(x.split()) + 0.1))
    data['token_2'] = data['Sentence'].apply(lambda x: log_or_no(token_chars_gt(x, 2)+0.1))
    data['token_6'] = data['Sentence'].apply(lambda x: log_or_no(token_chars_gt(x, 6)+0.1))
    data['token_7'] = data['Sentence'].apply(lambda x: log_or_no(token_chars_gt(x, 7)+0.1))
    data['token_8'] = data['Sentence'].apply(lambda x: log_or_no(token_chars_gt(x, 8)+0.1))
    data['token_10'] = data['Sentence'].apply(lambda x: log_or_no(token_chars_gt(x, 10)+0.1))

    data['type_token'] = data['Sentence'].apply(lambda x: log_or_no(len(list(set(x.split()))) / len(x.split()) + 0.1))
    data['carroll_ttr'] = data['Sentence'].apply(lambda x: log_or_no(len(list(set(x.split()))) / math.sqrt(2*len(x.split()))+0.1))
    
    data['num_common'] = data['Sentence_lemma'].apply(lambda x: log_or_no(sum([1 for y in x.split() if y in common_map])+0.1))
    regex = re.compile('[^a-zA-Z]')
    data['common_score'] = data['Sentence_lemma'].apply(lambda x: log_or_no(sum([common_map[regex.sub('', y)] for y in x.split() if regex.sub('', y) in common_map])+0.1))

    data['longest_word'] = data['Sentence'].apply(lambda x: log_or_no(len(max(x.split(), key=len))))
    data['commas'] = data['Sentence'].apply(lambda x: log_or_no(x.count(',')+0.1))
    data['parens'] = data['Sentence'].apply(lambda x: log_or_no(x.count('(')+0.1))
    data['digits'] = data['Sentence'].apply(lambda x: log_or_no(sum([1 for y in x if y.isdigit() == True])+0.1))
    data['quotes'] = data['Sentence'].apply(lambda x: log_or_no(sum([1 for y in x if y == "\'" or y == "\""])+0.1))
    data['avg_word_length'] = data['Sentence_proc'].apply(lambda x: log_or_no(sum([len(y.translate(str.maketrans('', '', string.punctuation))) for y in x.split() if len(y) > 0]) / len(x.split())))
    data['wordrank_score'] = data['Sentence_lemma'].apply(lambda x: log_or_no(sum([words[y.lower()] for y in x.split() if y.lower() in words and y.lower() not in STOP]) * len(x.split())))
    data['wordrank_score'] = data['wordrank_score'].apply(lambda x: log_or_no(x+1))

    for c in ['ADJ', 'ADP', 'ADV', 'AUX', 'NOUN', 'NUM', 'PRON','PROPN', 'VERB', 'X']:
        data[c+'_ratio'] = data['Sentence'].apply(lambda x: log_or_no(sum([1 for y in nlp(x) if y.pos_ == c]) / len(x.split())+0.1))

    for c in best_pos:
        data[c] = 0

    sentences = data['Sentence'].tolist()
    pos = []
    for s in sentences:
        temp = {}
        doc = nlp(s)
        counts = Counter([x.tag_ for x in doc])
        for p in pos_tags:
            if p in counts:
                temp[p] = counts[p]
            else:
                temp[p] = 0
        pos.append(temp)
    pos_df = pd.DataFrame(pos)
    
    for k in best_pos:
        temp = pos_df[k].tolist()
        data[k] = temp

    data['dep_length'] = data['Sentence_proc'].apply(lambda x: log_or_no(dep_parse_length(x)))
    data['ne_length'] = data['Sentence'].apply(lambda x: log_or_no(ne_length(x)+1))
    data['ne'] = data['Sentence'].apply(lambda x: log_or_no(ne(x)+1))
    data['l2_norm'] = data['Sentence_proc'].apply(lambda x: log_or_no(l2norm(x)))
    data['vec_exists'] = data['Sentence_lemma'].apply(lambda x: log_or_no(vec_exists(x)+0.1))

    data['syn_height'] = data['Sentence'].apply(lambda x: len(x) / syn_tree(x).height())
    data['leaves'] = data['Sentence'].apply(lambda x: len(syn_tree(x).leaves()))
    data['subtree'] = data['Sentence'].apply(lambda x: len([x for x in syn_tree(x).subtrees()]))
    data['leaf_distance'] = data['Sentence'].apply(lambda x: leaf_distance(x))

    data['tot_syl'] = data['Sentence'].apply(lambda x: sum([syllable(y) for y in x.split()]))
    data['avg_syl'] = data['Sentence'].apply(lambda x: sum([syllable(y) for y in x.split()]) / len(x.split()))
    
    data['flesch'] = data.apply(lambda x: max(-1 * (206.385 - (1.015 * len(x['Sentence'])) - (84.6 * x['avg_syl'])), 0), axis=1)
    
    data['single_syl'] = data['Sentence'].apply(lambda x: sum([1 for y in x.split() if syllable(y) == 1]))
    data['flesch_mod'] = data.apply(lambda x: max(-1*(1.599*x['single_syl'] - 1.015*len(x['Sentence']) - 31.517), 0), axis=1)
   
    data['hard_words'] = data['Sentence'].apply(lambda x: sum([1 for y in x.split() if syllable(y) > 2]))
    data['gunning_fog'] = data.apply(lambda x: 0.4 * (len(x['Sentence']) + x['hard_words']), axis=1)
    data['mod_smog'] = data['hard_words'].apply(lambda x: 3 + math.sqrt(x))
    
    data['mod_forcast'] = data['single_syl'].apply(lambda x: max(-1 * (20 - (x / 10)), 0))
    
    data['ari'] = data['Sentence'].apply(lambda x: max(4.71*(len(x) / len(x.split())) + 0.5*(len(x.split())) - 21.43, 0))
  
    data['easy_words'] = data['Sentence'].apply(lambda x: log_or_no(sum([1 for y in x.split() if syllable(y) <= 2]) / len(x.split())))
    data['linsear'] = data.apply(lambda x: max(linsear(x), 0), axis=1)

    # scrabble
    #data['new_scrabble_score'] = data['Sentence_lemma'].apply(lambda x: np.log(sum([get_scrabble_score_new(y) for y in x.split()]) / len(x.split())))
    #data['old_scrabble_score'] = data['Sentence_lemma'].apply(lambda x: np.log(sum([get_scrabble_score_old(y) for y in x.split()]) / len(x.split())))

    # wordnet stuff
    #data['synset_exists'] = data['Sentence_lemma'].apply(lambda x: len(get_synsets(x)))
    #data['synset_depth'] = data['Sentence_lemma'].apply(lambda x: get_max_depth(get_synsets(x)))
    #data['hyponyms'] = data['Sentence_lemma'].apply(lambda x: get_hypo(get_synsets(x)))
    #data['senses'] = data['Sentence_lemma'].apply(lambda x: get_senses(get_synsets(x)))
    #data['syn_def'] = data['Sentence_lemma'].apply(lambda x: get_def(get_synsets(x)))
    #data['avg_path'] = data['Sentence_lemma'].apply(lambda x: path(get_synsets(x)))

    return data

if __name__ == "__main__":

    VAL = False
    TEST = True

    if TEST == True:
        # open model, run on test data
        MODEL = "tumuch.pkl"
        with open(MODEL, 'rb') as f:
            model = pickle.load(f)
        test_data = pd.read_csv("test_set/part2_public.csv", index_col=[0])
        test_X = process(test_data)

        y_pred_test = model.predict(test_X.iloc[:,3:])
        test_X['MOS'] = y_pred_test
        save = test_X['MOS'].copy()

        save.to_csv("test_set/answer.csv")
    else:
        # search for existing training data with features, else create
        if Path("c_train.csv").is_file() == False:
            train = pd.read_csv("training_set.csv", index_col=[0]).iloc[:900, :]
            y = pd.read_csv("training_set.csv", index_col=[0]).iloc[:900, :]['MOS']
    
            train = train.drop(['MOS'], axis=1)
            X_new = process(train)

            X_new.to_csv("c_train.csv", index=False)
        else:
            X_new = pd.read_csv("c_train.csv").dropna(subset=['Sentence']).iloc[:900, :]
            y = pd.read_csv("training_set.csv", index_col=[0]).iloc[:900, :]['MOS']

        X_forreal = X_new.drop(['Sentence', 'Sentence_proc', 'Sentence_lemma'], axis=1).fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(X_forreal, y, test_size=0.1, random_state=42)

        kernel2 = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)

        # add noise to targets
        std = 1.06
        rng = np.random.RandomState(42)
        y_train = y_train + rng.normal(loc=0.0, scale=std, size=y_train.shape)

        regr = Pipeline(steps=[('scaler',StandardScaler(with_mean=False, with_std=False)), 
                            ('pca', PCA(n_components=20, svd_solver="randomized", random_state=42)),
                            ('regr', GaussianProcessRegressor(kernel=kernel2, alpha=std**2, random_state=123, n_restarts_optimizer=10))])
        
        params = {
            # enter grid search params here
        }

        # TUNING
        TUNE = False
        if TUNE:
            rkf = RepeatedKFold(n_splits=2, n_repeats=2)
            gs = GridSearchCV(regr, param_grid=params, n_jobs=-1, cv=rkf, scoring=mse_pos)
            search = gs.fit(X_train, y_train)
        
            print(search.best_score_)
            print(search.best_params_)

            regr = search.best_estimator_
        else:
            regr.fit(X_train, y_train)

        y_pred = regr.predict(X_test)

        print("RMSE: {}".format(mean_squared_error(y_test, y_pred, squared=False)))

        pickle.dump(regr, open("tumuch.pkl", 'wb'))

        # run during development phase
        if VAL == True:
            val = pd.read_csv("validation_set.csv", index_col=[0])
            new_val = process(val)

            y_pred_val = regr.predict(new_val.iloc[:,3:])
            new_val['MOS'] = y_pred_val
            save = new_val['MOS'].copy()

            save.to_csv("answer.csv")