# coding=utf-8
from __future__ import print_function
import sys,os
import json
import collections
import re
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.externals import joblib
from scipy import sparse
from zhon import hanzi
import string

reload(sys)
sys.setdefaultencoding('utf8')
import Queue as Q

def parser_char_for_word2(word):
    """
    parser char for word
    """
    if word.isdigit():
        word = word
    for i in range(len(word)):
        if word[i] >= u'\u4e00' and word[i] <= u'\u9fa5':
            word_out = " ".join(word)
            word_out = re.sub(" +", " ", word_out)
            return word_out
    return word


def parser_char_for_sentence(text):
    """
    parser char for text
    """
    
    words = text.strip().split()
    for i, word in enumerate(words):
        words[i] = parser_char_for_word2(word)
    return re.sub(" +", " ", ' '.join(words))
    
def inverted_index_generate(text):
    inverted_index = {}
    stops = []

    stops.extend(hanzi.punctuation)
    stops.extend(string.punctuation)
    stops.append("的".decode('utf8'))
    stops.append("吧".decode('utf8'))
    stops.append("了".decode('utf8'))
    
    for index, sentence in enumerate(text):    
        sentence = sentence.split()
        
        sentence = set(sentence)
        
        for word in sentence:
            if word in stops:
                continue
            if word in inverted_index:
                inverted_index[word].append(index)
            else:
                inverted_index[word] = [index, ]

    return inverted_index

def get_candidatelist(inverted_index, text):
    res = []
    for word in text.split():
        if word in inverted_index:
            res.extend(inverted_index[word])

    return list(set(res))
    
def get_candidatelist_bygoal(goal_inverted_index, goal1, goal2):
    res = []
    if goal1 in goal_inverted_index:
        res.extend(goal_inverted_index[goal1])
    if goal2 in goal_inverted_index:
        res.extend(goal_inverted_index[goal2])

    return list(set(res))

def prepare_candidate(session_file):
    text = []
    candidate_collection = []
    goal_inverted_index = {}
    index_for_goal = 0
    goal1 = None
    goal2 = None

    with open(session_file, 'r') as f:
        for i, line in enumerate(f):
            session = json.loads(line.strip(), encoding='utf-8', object_pairs_hook=collections.OrderedDict)
            goal1 = session["goal"][0][1]
            goal2 = session["goal"][0][2]
            if goal1 not in goal_inverted_index:
                goal_inverted_index[goal1] = []
            if goal2 not in goal_inverted_index:
                goal_inverted_index[goal2] = []
               
            if "conversation" in session:
                conversation = session["conversation"]
            elif "history" in session:
                if len(session["history"]) < 2:
                    continue
                conversation = session["history"]

            start = ' '.join(session["goal"][1])
            start = "##**" + start
            candidate_collection.append(start)
            text.append(parser_char_for_sentence(start))
            index_for_goal += 1
            
            for i in range(len(conversation)):
                candidate_collection.append(conversation[i])
           
                sentence = conversation[i]

                text.append(parser_char_for_sentence(sentence))
                goal_inverted_index[goal1].append(index_for_goal)
                goal_inverted_index[goal2].append(index_for_goal)
                index_for_goal += 1
                
            #candidate_collection.append('EOS')
            #text.append(parser_char_for_sentence('EOS'))

    # �����һ�仰�޳���
    text.pop()
    text.pop()
    goal_inverted_index[goal1].pop()
    goal_inverted_index[goal2].pop()
    goal_inverted_index[goal1].pop()
    goal_inverted_index[goal2].pop()

    inverted_index = inverted_index_generate(text)

    hv = HashingVectorizer(n_features=2**24, ngram_range=(2,2), analyzer='char', norm=None,lowercase=False)
    matrix = hv.transform(text)
    transformer = TfidfTransformer(smooth_idf=True)
    tfidf = transformer.fit_transform(matrix)

    joblib.dump(hv, "HashingVectorizer.m")
    joblib.dump(transformer, "TfidfTransformer.m")
    sparse.save_npz('tfidf.npz',tfidf)
    with open('inverted_index.json', 'wb') as json_file:
        json.dump(inverted_index, json_file)
    with open('goal_inverted_index.json', 'wb') as json_file:
        json.dump(goal_inverted_index, json_file)
    with open('candidate_collection.json', 'wb') as json_file:
        json.dump(candidate_collection, json_file)

    return hv, transformer, tfidf, inverted_index, goal_inverted_index, text


def get_candidate(hv, transformer, tfidf, inverted_index,text, conversation):
    sentence = conversation[-1]
    sentence = [parser_char_for_sentence(sentence)]
    last = hv.transform(sentence)
    last_tfidf = transformer.transform(last)
    result = []
    max_index = len(text)-1

    candidate_list = get_candidatelist(inverted_index, sentence[0])
    cos = cosine_similarity(tfidf[candidate_list], last_tfidf)
    sorted_cos = sorted(enumerate(cos), key=lambda x:-x[1])
    result_index = [candidate_list[candi[0]] for candi in sorted_cos[:10]]
    result = [text[index+1] for index in result_index if index < max_index]

    return result
    
def main():
    hv, transformer, tfidf, inverted_index, goal_inverted_index, text = prepare_candidate(sys.argv[1])
    result = get_candidate(hv, transformer, tfidf, inverted_index,text, ["80后 真 的 是 利害 啊 ， 读 大学 利害 ， 当 演员 也 利害 。".decode('utf8')])
    print(result)
    result = get_candidate(hv, transformer, tfidf, inverted_index,text, ["听 名字 就 感觉 很好看 ， 有空 我 去 看看".decode('utf8')])
    print(result)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
