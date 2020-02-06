import os
import glob
import multiprocessing

from sklearn.pipeline import Pipeline
import numpy as np

from gurih.data.normalizer import AudioNormalizer
from gurih.features.extractor import MFCCFeatureExtractor
from gurih.utils import batch, validate_nonavailability

def extraction_pipeline_batch(id, batch, pipeline):
    """
    Perform extracting in batches.
    
    Parameters
    ----------
    id : str
        ID of worker

    batch : list of str
        List of .json aligned files

    pipeline : sklearn.pipeline
        Pipeline used to extract the features
    """
    for mp3 in batch:
        pipeline.fit_transform([mp3])
        
    print(f"Worker {id} done.")

    return 0

if __name__ == '__main__':
    mp3s = glob.glob("*.mp3")

    # Checking the availability of the npz for the corresponding mp3s
    file_availability_dict = validate_nonavailability(mp3s, "npz")

    norm_feature_extractor = Pipeline(
        steps = [
            ("normalizer", AudioNormalizer(encode=False)),
            ("mfcc_feature_extractor", MFCCFeatureExtractor(append_delta=True, low_memory=True, write_output=True))
        ]
    )

    # Find only the mp3s that currently do not have the corresponding npz
    non_extracted_mp3s = [f"{k.replace('npz', 'mp3')}" for k, v in file_availability_dict.items() if v]
    print(f"Currently, there are {len(non_extracted_mp3s)} mp3s left to be extracted")
    
    cpus = os.cpu_count()
    batch_size = int(len(mp3s) / (cpus - 1))
    batches = batch(non_extracted_mp3s, batch_size)

    # Spawn jobs
    jobs = []
    for i, b in zip(range(cpus - 1), batches): # don't use all cores
        p = multiprocessing.Process(target=extraction_pipeline_batch, args=[i, b, norm_feature_extractor])
        jobs.append(p)
        p.start()