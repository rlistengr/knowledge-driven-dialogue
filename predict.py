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
"""Load checkpoint of running classifier to do prediction and save inference model."""

import os
import time
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.framework as framework

import data_provider as reader

import multiprocessing
from train import create_model
from args import base_parser
from args import print_arguments
from utils import init_pretraining_params


def main(args):

    task_name = args.task_name.lower()
    processor = reader.MatchProcessor(data_dir=args.data_dir,
                                      task_name=task_name,
                                      vocab_path=args.vocab_path,
                                      max_seq_len=args.max_seq_len,
                                      do_lower_case=args.do_lower_case)
    
    num_labels = len(processor.get_labels())
    infer_data_generator = processor.data_generator(
        batch_size=args.batch_size,
        phase='test',
        epoch=1,
        shuffle=False)
    num_test_examples = processor.get_num_examples(phase='test')
    main_program = fluid.default_main_program()

    feed_order, loss, probs, accuracy, num_seqs = create_model(
                args,
                num_labels=num_labels,
                is_prediction=True)
    
    if args.use_cuda: 
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

    exe = fluid.Executor(place)
    exe.run(framework.default_startup_program())

    if args.init_checkpoint: 
        init_pretraining_params(exe, args.init_checkpoint, main_program)

    feed_list = [
        main_program.global_block().var(var_name) for var_name in feed_order
        ]
    feeder = fluid.DataFeeder(feed_list, place)

    for batch_id, data in enumerate(infer_data_generator()): 
        results = exe.run(
                fetch_list=[probs],
                feed=feeder.feed(data),
                return_numpy=True)
        for elem in results[0]: 
            print(elem[1])




if __name__ == '__main__':
    args = base_parser()
    print_arguments(args)
    main(args)
