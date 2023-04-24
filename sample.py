from typing import Any, Dict, List, Union
from .base import Pipeline
import numpy as np
import os
from os.path import join
import multiprocessing as mp
from abc import abstractmethod
from ..configs.data_schema import *
from ..utils_dev.parallel_utils import set_cpu_affinity, get_cpu_affinity
import torch
import uuid
import json
# logger = logging.get_logger(__name__)


class EncDecTranscriptionPipeline(Pipeline):
    """
    Encoder-decoder based transcription pipeline, used for:
    1. text transcription with top-n candidates, with character level confidence
    2. target transcription with character level confidence
    3. encoder-decoder decoupling, with encoder savestate, to enable campability 
       of decoding multiple times with the same encoder state


    Example:

    ```python
    >>> in[0]
    from allston import DummyEDTranscriptionPipline

    # production
    transcriber = DummyEDTranscriptionPipline(model="homework/math") 

    # dev (you can pass in any extra parameters using kwargs)
    transcriber = DummyEDTranscriptionPipline(
        model="homework/math",
        ckpt_path = "/home/ubuntu/checkpoints/seg_0000.pth",
        batch_size=16,
        n_candidates=5,
        ...
    )

    # inference, check allston.configs.data_schema for all schemas
    encodeded_features = transcriber.encode(list_of_images) 
    top5_results = transcriber.decode(encodeded_features)
    targeted_results = transcriber.decode(encodeded_features,targets=list_of_targets)

    ```
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = "enc-dec-transcription"
        # pass in task name



    @abstractmethod
    def encode(self, images, **kwargs):
        """
        Perform segmentation (detect masks & classes) in the image(s) passed as inputs.

        Args:
            images (List[numpy.Image]):
            segmented image, colored opencv image with shape (H,W,3), background is white, foreground is black


        Return:
            @mason add when finished
        """
        pass

    @abstractmethod
    def decode(self, encodeded_features, targets=None, **kwargs):
        """
        Perform segmentation (detect masks & classes) in the image(s) passed as inputs.
        
        Args:
            images (List[numpy.Image]):
                segmented image, colored opencv image with shape (H,W,3), background is white, foreground is black
            encodeded_features (): 
        Return:
            list of dictionaries with the following keys:
            TranscriptionDecoderResult = {
                "sequences": [str],
                "sequence_confidence": [float],
                "tokens": [[str]],
                "token_confidences": [[float]],
                "targeted": bool
            }
        """
        pass

    def __call__(self, images, targets=None, **kwargs):
        """
        Perform 

        Args:
            images (List[numpy.Image]):
                segmented image, colored opencv image with shape (H,W,3), background is white, foreground is black
            targets (List[str]):
                list of target strings, optional, if provided, will perform targeted decoding

        Return:
            list of dictionaries with the following keys:
            TranscriptionDecoderResult = {
                "sequences": [str],
                "sequence_confidence": [float],
                "tokens": [[str]],
                "token_confidences": [[float]],
                "targeted": bool
            }
        """
        encodeded_features = self.encode(images, **kwargs)
        if targets is None:
            return self.decode(encodeded_features, **kwargs)
        else:
            return self.decode(encodeded_features, targets=targets, **kwargs)

    def run_worker(self, cpu_id, data_batches, output_dir):
        """
        multiprocessing worker function, will be called by the multiprocessing pool,
        saves whole batch of results to disk, and returns the path to the saved file
        save format {"paths": [str], "tr_results": [TranscriptionDecoderResult]}

        Args:
            images (List[numpy.Image]):
                segmented image, colored opencv image with shape (H,W,3), background is white, foreground is black
            targets (List[str]):
                list of target strings, optional, if provided, will perform targeted decoding

        Return:
            list of dictionaries with the following keys:
            TranscriptionDecoderResult = {
                "sequences": [str],
                "sequence_confidence": [float],
                "tokens": [[str]],
                "token_confidences": [[float]],
                "targeted": bool
            }
        r
        """
        
        set_cpu_affinity(cpu_id)
        num_threads = 1
        if torch.get_num_threads() != num_threads:
            torch.set_num_threads(num_threads)
        interp_threads = 1
        if torch.get_num_interop_threads() != interp_threads:
            torch.set_num_interop_threads(interp_threads)

        p = mp.current_process()

        images = [data['img'] for data in data_batches]
        paths = [data['path'] for data in data_batches]
        encodeded_features = self.encode(images)
        tr_results = self.decode(encodeded_features)
        for tr_result in tr_results:
            del tr_result['sequence_trees']

        result_json = {'paths':paths,'tr_results':tr_results}
        batch_id = str(uuid.uuid4())
        with open(join(output_dir,batch_id+".json"), 'w') as fp:
            json.dump(result_json, fp, sort_keys=True, indent=0, ensure_ascii=False)

        p.close()


    





