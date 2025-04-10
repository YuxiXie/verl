# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Preprocess the OpenMathInstruct2 dataset to parquet format
"""

import os
import random
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string, is_equiv

random.seed(42)


def extract_solution(solution_str):
    rst = last_boxed_only_string(solution_str)
    return remove_boxed(rst) if rst is not None else None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/math')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'nvidia/OpenMathInstruct-2'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    
    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
    
    # train_5M_dataset = datasets.load_dataset(data_source, split='train_5M', trust_remote_code=True)
    train_1M_dataset = datasets.load_dataset(data_source, split='train_1M', trust_remote_code=True)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(x):
            idx, example = x
            question = example.pop('problem')
            answer = example.pop('generated_solution')
            ground_truth = example.pop('expected_answer')
            if question is None or answer is None or not question.strip() or not answer.strip() or ground_truth is None or not ground_truth.strip(): return None

            solution = extract_solution(answer)
            if not is_equiv(ground_truth, solution): return None
            
            if random.random() > 0.1: return None
            
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question + ' ' + instruction_following
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "demo": answer,
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn
    
    # train_5M_dataset = train_5M_dataset.map(function=make_map_fn('train_5M'), with_indices=True)
    train_1M_dataset = map(make_map_fn('train_1M'), enumerate(train_1M_dataset))
    # train_5M_dataset = datasets.Dataset.from_list([x for x in train_5M_dataset if x is not None])
    train_1M_dataset = datasets.Dataset.from_list([x for x in train_1M_dataset if x is not None and len(x) > 1])

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # train_5M_dataset.to_parquet(os.path.join(local_dir, 'train_5M.parquet'))
    train_1M_dataset.to_parquet(os.path.join(local_dir, 'train_100K.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
