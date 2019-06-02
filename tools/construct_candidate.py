#!/usr/bin/env python
# -*- coding: utf-8 -*-
######################################################################
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
######################################################################
"""
File: construct_candidate.py
"""

from __future__ import print_function
import sys
sys.path.append("./")
import json
import collections

reload(sys)
sys.setdefaultencoding('utf8')

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.externals import joblib
from scipy import sparse
from tools.prepare_candidate import parser_char_for_sentence
from tools.prepare_candidate import get_candidatelist
from tools.prepare_candidate import get_candidatelist_bygoal
import random

_get_candidate_for_conversation_init = 0
_hv = None
_transformer = None
_tfidf = None
_inverted_index = None
_text = None
_goal_inverted_index = None


def get_candidate_for_conversation(conversation, for_predict, for_dev):
    """
    get candidate for conversation
    !!! you have to reimplement the function yourself !!!
    """
    global _get_candidate_for_conversation_init
    global _hv
    global _transformer
    global _tfidf
    global _inverted_index
    global _text
    global _goal_inverted_index
    if 0 == _get_candidate_for_conversation_init:
        _hv = joblib.load("./data/resource/HashingVectorizer.m")
        _transformer = joblib.load("./data/resource/TfidfTransformer.m")
        _tfidf = sparse.load_npz("./data/resource/tfidf.npz")
        with open("./data/resource/inverted_index.json",'rb') as json_file:
            _inverted_index = json.load(json_file)
        with open("./data/resource/goal_inverted_index.json",'rb') as json_file:
            _goal_inverted_index = json.load(json_file)
        with open("./data/resource/candidate_collection.json",'rb') as json_file:
            _text = json.load(json_file)
        _get_candidate_for_conversation_init = 1

    result = []
    result_tmp = []

    if for_predict:
        # 先利用前一句话获得候选回答
        history = conversation["history"]
        turns = len(history)
        if turns == 0:
            sentence = ' '.join(conversation["goal"][1])
            sentence = "##**" + sentence
        else:
            sentence = history[turns-1]
            
        sentence = [parser_char_for_sentence(sentence)]
        last = _hv.transform(sentence)
        last_tfidf = _transformer.transform(last)
        max_index = len(_text)-1
        candidate_list = []
        
        candidate_list = get_candidatelist(_inverted_index, sentence[0])
        
        #if len(candidate_list) > 100:
        #    candidate_list = random.sample(candidate_list, 100)
        if len(candidate_list) == 0:
            candidate_list.extend(random.sample(range(0, max_index), 100))
        
        if len(candidate_list) <= 7:
            result_tmp.extend([index+1 for index in candidate_list])
            # result.extend(candidate_list)
        else:
            cos = cosine_similarity(_tfidf[candidate_list], last_tfidf)
            sorted_cos = sorted(enumerate(cos), key=lambda x:-x[1])
            result_index = [candidate_list[candi[0]] for candi in sorted_cos[:7]]
            result_tmp = [index+1 for index in result_index]
            # result = [_text[index+1] for index in result_index]
        
        # 如果有前前一句话的话，再获取7个候选回答
        if turns > 1:
            sentence = history[turns-2]
            sentence = [parser_char_for_sentence(sentence)]
            last2 = _hv.transform(sentence)
            last2_tfidf = _transformer.transform(last2)
            max_index = len(_text)-2
            candidate_list = []
            
            candidate_list = get_candidatelist(_inverted_index, sentence[0])
            
            #if len(candidate_list) > 100:
            #    candidate_list = random.sample(candidate_list, 100)
            if len(candidate_list) == 0:
                candidate_list.extend(random.sample(range(0, max_index), 100))

            if len(candidate_list) <= 7:
                result_tmp.extend([index+2 for index in candidate_list])
               # result.extend(candidate_list)
            else:
                cos = cosine_similarity(_tfidf[candidate_list], last2_tfidf)
                sorted_cos = sorted(enumerate(cos), key=lambda x:-x[1])
                result_index = [candidate_list[candi[0]] for candi in sorted_cos[:7]]
                result_tmp.extend([index+2 for index in result_index])
                # result.extend([_text[index+2] for index in result_index])
        
        # 通过主题选取7个候选回答
        goal_candidate_list = get_candidatelist_bygoal(_goal_inverted_index, conversation['goal'][0][1], conversation['goal'][0][2])
        if len(goal_candidate_list) > 0:
            if len(goal_candidate_list) < 7:
                result_tmp.extend([index+1 for index in goal_candidate_list])
                # result.extend(goal_candidate_list)
            else:
                cos = cosine_similarity(_tfidf[goal_candidate_list], last_tfidf)
                sorted_cos = sorted(enumerate(cos), key=lambda x:-x[1])
                result_index = [goal_candidate_list[candi[0]] for candi in sorted_cos[:7]]
                result_tmp.extend([index+1 for index in result_index])
                # result.extend([_text[index+1] for index in result_index])  

        result_tmp = list(set(result_tmp))
        result = [_text[index] for index in result_tmp]

        if for_dev and "response" in conversation:
            try:
                result.remove(conversation["response"])
            except ValueError:
                pass
        

    else:
        result_tmp = random.sample(range(0, len(_text)), 10)
        
        result = [_text[index] for index in result_tmp]

    return result


def construct_candidate_for_corpus(corpus_file, candidate_file, for_predict, for_dev=0):
    """
    construct candidate for corpus

    case of data in corpus_file:
    {
        "goal": [["START", "休 · 劳瑞", "蕾切儿 · 哈伍德"]],
        "knowledge": [["休 · 劳瑞", "评论", "完美 的 男人"]],
        "history": ["你 对 明星 有没有 到 迷恋 的 程度 呢 ？",
                    "一般 吧 ， 毕竟 年纪 不 小 了 ， 只是 追星 而已 。"]
    }

    case of data in candidate_file:
    {
        "goal": [["START", "休 · 劳瑞", "蕾切儿 · 哈伍德"]],
        "knowledge": [["休 · 劳瑞", "评论", "完美 的 男人"]],
        "history": ["你 对 明星 有没有 到 迷恋 的 程度 呢 ？",
                    "一般 吧 ， 毕竟 年纪 不 小 了 ， 只是 追星 而已 。"],
        "candidate": ["我 说 的 是 休 · 劳瑞 。",
                      "我 说 的 是 休 · 劳瑞 。"]
    }
    """
    fout_text = open(candidate_file, 'w')
    with open(corpus_file, 'r') as f:
        for i, line in enumerate(f):
            conversation = json.loads(line.strip(), encoding="utf-8", \
                                 object_pairs_hook=collections.OrderedDict)
            candidates = get_candidate_for_conversation(conversation, for_predict,
                                                        for_dev=for_dev)
                
            conversation["candidate"] = candidates

            conversation = json.dumps(conversation, ensure_ascii=False, encoding="utf-8")
            fout_text.write(conversation + "\n")
            break

    fout_text.close()


def main():
    """
    main
    """
    construct_candidate_for_corpus(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
