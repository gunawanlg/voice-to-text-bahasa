"""
Given .json aligned files. Split it to each fragment of .mp3 and .txt.
Script written to utilize multiprocessing for faster processing.

Author: Gunawan Lumban Gaol
Date: February 05, 2020

Copyright 2020 Gunawan Lumban Gaol

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import glob
import argparse
import multiprocessing

from gurih.data.splitter import AeneasSplitter
from gurih.utils import batch

def split_batch(id, batch, splitter):
    """
    Perform splitting in batches.

    Parameters
    ----------
    id : str
        id of worker
    batch : list of str
        list of .json aligned files
    splitter : AeneasSplitter
        class to split audio into fragments
    """
    print(f"Start worker {id}.")
    for json in batch:
        fragments = splitter.load(json)
        splitter.split_and_write(fragments)
    
    print(f"Worker {id} done.")

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='2.0-glg-split_mp',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input_dir', default='input/',
                        help='Input directory of .json files.')
    parser.add_argument('--output_dir', default='output/',
                        help='Output directory of .mp3 and .txt files.')

    args = parser.parse_args()

    # Set splitter
    input_dir = args.input_dir
    output_dir = args.output_dir
    splitter = AeneasSplitter(input_dir=input_dir, output_dir=output_dir)

    # Create batches
    aligned_jsons = glob.glob(input_dir+"*.json")
    aligned_jsons = [os.path.basename(path) for path in aligned_jsons]
    print(f"Processing {len(aligned_jsons)} json files.")
    
    cpus = os.cpu_count()
    batch_size = int(len(aligned_jsons) / (cpus - 1))
    batches = batch(aligned_jsons, batch_size)

    # Spawn jobs
    jobs = []
    for i, b in zip(range(cpus - 1), batches): # don't use all cores
        p = multiprocessing.Process(target=split_batch, args=[i, b, splitter])
        jobs.append(p)
        p.start()