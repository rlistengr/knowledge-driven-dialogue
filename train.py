#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
# ecoding=utf8

import os
import time
import numpy as np
import multiprocessing

import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework
from paddle.fluid.executor import Executor

import data_provider as reader
from model import RetrievalModel
from args import base_parser
from args import print_arguments
from sklearn.metrics import classification_report

def create_model(args, num_labels, is_prediction=False):
    context_ids = fluid.layers.data(name='context_ids', shape=[-1, args.max_seq_len, 1], dtype='int64', lod_level=0)
    context_pos_ids = fluid.layers.data(name='context_pos_ids', shape=[-1, args.max_seq_len, 1], dtype='int64', lod_level=0)
    context_segment_ids = fluid.layers.data(name='context_segment_ids', shape=[-1, args.max_seq_len, 1], dtype='int64', lod_level=0)
    context_attn_mask = fluid.layers.data(name='context_attn_mask', shape=[-1, args.max_seq_len, args.max_seq_len], dtype='float', lod_level=0)
    labels = fluid.layers.data(name='labels', shape=[1], dtype='int64', lod_level=0)
    indexs = fluid.layers.data(name='indexs', shape=[2], dtype='float32', lod_level=0)
    weight111 = fluid.layers.data(name='weight111', shape=[2], dtype='float32', lod_level=0)
    #weight_ones = fluid.layers.data(name='weight_ones', shape=[2], dtype='int', lod_level=0)
    onesforlog = fluid.layers.data(name='onesforlog', shape=[2], dtype='float32', lod_level=0)
    context_next_sent_index = fluid.layers.data(name='context_next_sent_index', shape=[1], dtype='int64', lod_level=0)

    if "kn" in args.task_name:
        kn_ids = fluid.layers.data(name='kn_ids', shape=[1], dtype='int64', lod_level=1)
        feed_order = ["context_ids", "context_pos_ids", "context_segment_ids", "context_attn_mask", "kn_ids", "labels", "indexs", "weight111", "onesforlog", "context_next_sent_index"]
    else:
        kn_ids = None
        feed_order = ["context_ids", "context_pos_ids", "context_segment_ids", "context_attn_mask", "labels", "context_next_sent_index"]

    if not is_prediction:
        dropout_prob = 0.4
        attention_dropout = 0.4
        prepostprocess_dropout = 0.4
    else:
        dropout_prob = 0.0
        attention_dropout = 0.0
        prepostprocess_dropout = 0.0

    retrieval_model = RetrievalModel(
        context_ids=context_ids,
        context_pos_ids=context_pos_ids,
        context_segment_ids=context_segment_ids,
        context_attn_mask=context_attn_mask,
        kn_ids=kn_ids,
        emb_size=512,
        n_layer=4,
        n_head=8,
        voc_size=args.voc_size,
        max_position_seq_len=args.max_seq_len,
        hidden_act="gelu",
        attention_dropout=attention_dropout,
        prepostprocess_dropout=prepostprocess_dropout)

    context_cls = retrieval_model.get_context_output(context_next_sent_index, args.task_name)
    context_cls = fluid.layers.dropout(
        x=context_cls,
        dropout_prob=dropout_prob,
        dropout_implementation="upscale_in_train")

    cls_feats = context_cls
    
    logits1 = fluid.layers.fc(
        input=cls_feats,
        size=512,
        act="relu",
        param_attr=fluid.ParamAttr(
            name="logits1_cls_out_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.2)),
        bias_attr=fluid.ParamAttr(
            name="logits1_cls_out_b", initializer=fluid.initializer.Constant(0.)))
    
    #logits11 = fluid.layers.square(logits1)
    #logits1111 = fluid.layers.pow(logits1, 3.0)
    #logits111 = fluid.layers.concat(input=[logits1, logits11,logits1111], axis=1)
    
    logits2 = fluid.layers.fc(
        input=logits1,
        size=512,
        act="relu",
        param_attr=fluid.ParamAttr(
            name="logits2_cls_out_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.2)),
        bias_attr=fluid.ParamAttr(
            name="logits2_cls_out_b", initializer=fluid.initializer.Constant(0.)))


    logits = fluid.layers.fc(
        input=logits2,
        size=num_labels,
        act=None,
        param_attr=fluid.ParamAttr(
            name="cls_out_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.2)),
        bias_attr=fluid.ParamAttr(
            name="cls_out_b", initializer=fluid.initializer.Constant(0.)))

    #ce_loss, predict = fluid.layers.softmax_with_cross_entropy(
    #    logits=logits, label=labels, return_softmax=True)
    #mul = labels * 0.7
    #labels_loss_tmp = ce_loss * mul
    #ce_loss_ = ce_loss + labels_loss_tmp
    
    predict = fluid.layers.softmax(input=logits)
    softmax = fluid.layers.reshape(predict, [-1])
    indexs_ = fluid.layers.reshape(indexs, [-1])

    # prob = fluid.layers.gather(softmax, indexs2)
    prob = softmax * indexs_
    #weight_ones = fluid.layers.ones(shape=[200], dtype="float")
    weight_ = indexs_ - prob
    weight = fluid.layers.pow(weight_, 2.0)
    prob_ = prob + fluid.layers.reshape(onesforlog, [-1])
    ce_loss = weight * fluid.layers.log(prob_) * (-1)
    #mul = fluid.layers.reshape(weight111 * (100.0), [-1])
    #labels_loss_tmp = ce_loss * mul
    #ce_loss_ = ce_loss + labels_loss_tmp

    loss = fluid.layers.reduce_mean(input=ce_loss)

    num_seqs = fluid.layers.create_tensor(dtype='int64')
    accuracy = fluid.layers.accuracy(input=predict, label=labels, total=num_seqs)

    loss.persistable = True
    predict.persistable = True
    accuracy.persistable = True
    num_seqs.persistable = True

    return feed_order, loss, predict, accuracy, num_seqs, labels, softmax, prob, indexs_,logits


def main(args):

    task_name = args.task_name.lower()
    processor = reader.MatchProcessor(data_dir=args.data_dir,
                                      task_name=task_name,
                                      vocab_path=args.vocab_path,
                                      max_seq_len=args.max_seq_len,
                                      do_lower_case=args.do_lower_case)

    args.voc_size = len(open(args.vocab_path, 'r').readlines())
    num_labels = len(processor.get_labels())
    train_data_generator = processor.data_generator(
        batch_size=args.batch_size,
        phase='train',
        epoch=args.epoch,
        shuffle=True)
    num_train_examples = processor.get_num_examples(phase='train')
    dev_data_generator = processor.data_generator(
        batch_size=args.batch_size,
        phase='dev',
        epoch=1,
        shuffle=False)
    num_dev_examples = processor.get_num_examples(phase='dev')

    if args.use_cuda:
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

    max_train_steps = args.epoch * num_train_examples // args.batch_size
    warmup_steps = int(max_train_steps * args.warmup_proportion)

    train_program = fluid.Program()
    train_startup = fluid.Program()
    with fluid.program_guard(train_program, train_startup):
        with fluid.unique_name.guard():
            feed_order, loss, predict, accuracy, num_seqs, labels, softmax, prob, indexs1,logits = \
                    create_model(args, num_labels, \
                    is_prediction=False)
            lr_decay = fluid.layers.learning_rate_scheduler.noam_decay(256, warmup_steps)
            with fluid.default_main_program()._lr_schedule_guard():
                learning_rate = lr_decay * args.learning_rate
            optimizer = fluid.optimizer.Adam(
                learning_rate=learning_rate)
            optimizer.minimize(loss)

    test_program = fluid.Program()
    test_startup = fluid.Program()
    with fluid.program_guard(test_program, test_startup):
        with fluid.unique_name.guard():
            feed_order, loss, predict, accuracy, num_seqs, labels, softmax, prob, indexs1,logits = \
                    create_model(args, num_labels, \
                    is_prediction=True)
    test_program = test_program.clone(for_test=True)

    exe = Executor(place)
    exe.run(train_startup)
    exe.run(test_startup)

    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = dev_count

    train_exe = fluid.ParallelExecutor(
        use_cuda=args.use_cuda,
        loss_name=loss.name,
        exec_strategy=exec_strategy,
        main_program=train_program)


    test_exe = fluid.ParallelExecutor(
        use_cuda=args.use_cuda,
        main_program=test_program,
        share_vars_from=train_exe)

    feed_list = [
        train_program.global_block().var(var_name) for var_name in feed_order
    ]
    feeder = fluid.DataFeeder(feed_list, place)

    time_begin = time.time()
    total_cost, total_acc, total_num_seqs = [], [], []
    tmp11 = []
    tmp22 = []

    for batch_id, data in enumerate(train_data_generator()):
        fetch_outs = train_exe.run(
                    feed=feeder.feed(data),
                    fetch_list=[loss.name, accuracy.name, num_seqs.name, predict.name, labels.name, softmax.name,logits.name])
        avg_loss = fetch_outs[0]
        avg_acc = fetch_outs[1]
        cur_num_seqs = fetch_outs[2]
        total_cost.extend(avg_loss * cur_num_seqs)
        total_acc.extend(avg_acc * cur_num_seqs)
        total_num_seqs.extend(cur_num_seqs)
        results1 = fetch_outs[3]
        act1 = fetch_outs[4]
        for index in range(len(results1)):
            if results1[index][0] > results1[index][1]:
                tmp11.append(0)
            else:
                tmp11.append(1)
            tmp22.append(act1[index])

        if batch_id % args.skip_steps == 0:
            print(fetch_outs[5][0:3])
            #print(fetch_outs[6])
            print(fetch_outs[6][0:3])
            print(classification_report(tmp22, tmp11))
            tmp11 = []
            tmp22 = []
            
            time_end = time.time()
            used_time = time_end - time_begin
            current_example, current_epoch = processor.get_train_progress()
            print("epoch: %d, progress: %d/%d, step: %d, ave loss: %f, "
                "ave acc: %f, speed: %f steps/s" %
                (current_epoch, current_example, num_train_examples,
                batch_id, np.sum(total_cost) / np.sum(total_num_seqs),
                np.sum(total_acc) / np.sum(total_num_seqs),
                args.skip_steps / used_time))
            time_begin = time.time()
            total_cost, total_acc, total_num_seqs = [], [], []

        if batch_id % args.validation_steps == 0:
            total_dev_cost, total_dev_acc, total_dev_num_seqs = [], [], []
            tmp1 = []
            tmp2 = []
            for dev_id, dev_data in enumerate(dev_data_generator()):
                fetch_outs = test_exe.run(
                    feed=feeder.feed(dev_data),
                    fetch_list=[loss.name, accuracy.name, num_seqs.name, predict.name, labels.name])
                avg_dev_loss = fetch_outs[0]
                avg_dev_acc = fetch_outs[1]
                cur_dev_num_seqs = fetch_outs[2]
                results = fetch_outs[3]
                act = fetch_outs[4]

                for index in range(len(results)):
                    if results[index][0] > results[index][1]:
                        tmp1.append(0)
                    else:
                        tmp1.append(1)
                    tmp2.append(act[index])

                total_dev_cost.extend(avg_dev_loss * cur_dev_num_seqs)
                total_dev_acc.extend(avg_dev_acc * cur_dev_num_seqs)
                total_dev_num_seqs.extend(cur_dev_num_seqs)
                
            print(classification_report(tmp2, tmp1))
            print("valid eval: ave loss: %f, ave acc: %f" %
                 (np.sum(total_dev_cost) / np.sum(total_dev_num_seqs),
                  np.sum(total_dev_acc) / np.sum(total_dev_num_seqs)))
            total_dev_cost, total_dev_acc, total_dev_num_seqs = [], [], []

        # if batch_id % args.save_steps == 0:
    model_path = os.path.join(args.checkpoints, str(batch_id))
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    fluid.io.save_persistables(
        executor=exe,
        dirname=model_path,
        main_program=train_program)
        
        
    # predict
    print ("=================for predict===================")
    infer_data_generator = processor.data_generator(
        batch_size=args.batch_size,
        phase='test',
        epoch=1,
        shuffle=False)
    for batch_id, data in enumerate(infer_data_generator()): 
        results = test_exe.run(
                fetch_list=[predict.name],
                feed=feeder.feed(data),
                return_numpy=True)
        for elem in results[0]: 
            print(elem[1])
            
    print ("=================for dev===================")
    infer_data_generator2 = processor.data_generator(
        batch_size=args.batch_size,
        phase='dev',
        epoch=1,
        shuffle=False)
    for batch_id, data in enumerate(infer_data_generator2()): 
        results = test_exe.run(
                fetch_list=[predict.name],
                feed=feeder.feed(data),
                return_numpy=True)
        for elem in results[0]: 
            print(elem[1])


if __name__ == '__main__':
    print("==========start=============")
    dev_count = fluid.core.get_cuda_device_count()
    print("dev_count:{}".format(dev_count))
    print(os.environ)
    print("=======================")
    args = base_parser()

    args.task_name = 'match_kn_gene'
    args.use_cuda = True
    args.batch_size = 100
    args.data_dir = '/root/paddlejob/workspace/train_data/datasets/data6386'
    args.vocab_path = './dict/gene.dict'
    args.checkpoints = '/root/paddlejob/workspace/output'
    args.save_steps = 15000
    args.weight_decay = 0.01
    args.warmup_proportion = 0.1
    args.validation_steps = 5000
    args.skip_steps = 100
    args.learning_rate = 0.05
    args.epoch = 1
    args.max_seq_len = 128
    
    print_arguments(args)
    main(args)
