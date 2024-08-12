#!/usr/bin/env python
# encoding: utf-8
#
# Copyright 2022 Spotify AB
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

import argparse
import logging
import os
import os.path as op
import random
import time
from typing import List, Tuple, Optional

import apache_beam as beam
import mirdata

from basic_pitch.dataset import commandline, pipeline

DAGSTUHL_DIR = "dagstuhl_choirset"  # "DagstuhlChoirSet"


class DagstuhlInvalidTracks(beam.DoFn):
    DOWNLOAD_ATTRIBUTES = ["audio_lrx_path", "score_path", "f0_crepe_lrx_path" ]

    def __init__(self, source: str):
        self.source = source

    def setup(self):
        import apache_beam as beam
        import os
        import mirdata

        self.dagstuhl_remote = mirdata.initialize("dagstuhl_choirset", data_home=os.path.join(self.source, DAGSTUHL_DIR))
        self.filesystem = beam.io.filesystems.FileSystems()

    def process(self, element: Tuple[str, str]):
        import tempfile

        import apache_beam as beam
        import sox
        import soundfile as sf

        from basic_pitch.constants import (
            AUDIO_N_CHANNELS,
            AUDIO_SAMPLE_RATE,
        )

        track_id, split = element
        if split == "omitted":
            return None

        logging.info(f"Processing (track_id, split): ({track_id}, {split})")

        track_remote = self.dagstuhl_remote.track(track_id)

        with tempfile.TemporaryDirectory() as local_tmp_dir:
            dagstuhl_local = mirdata.initialize("dagstuhl_choirset", data_home=local_tmp_dir)
            track_local = dagstuhl_local.track(track_id)

            for attr in self.DOWNLOAD_ATTRIBUTES:
                source = getattr(track_remote, attr)
                dest = getattr(track_local, attr)
                if not dest:
                    return None
                logging.info(f"Downloading {attr} from {source} to {dest}")
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with self.filesystem.open(source) as s, open(dest, "wb") as d:
                    d.write(s.read())

            local_wav_path = "{}_tmp.wav".format(track_local.audio_lrx_path)
            tfm = sox.Transformer()
            tfm.rate(AUDIO_SAMPLE_RATE)
            tfm.channels(AUDIO_N_CHANNELS)
            
            try:
                tfm.build(track_local.audio_lrx_path, local_wav_path)
            except Exception as e:
                logging.info(f"Could not process {local_wav_path}. Exception: {e}")
                return None
            
            # wav files are not always PCM, so we convert them to PCM
            try:
                data, samplerate = sf.read(local_wav_path)    
                sf.write(local_wav_path, data, samplerate, subtype='PCM_16')
            except Exception as e:
                logging.info(f"Could not convert to PCM {local_wav_path}. Exception: {e}")
                return None
            
            # if there are no notes, skip this track
            if track_local.score is None or len(track_local.score.intervals) == 0:
                return None

            ##If return None skip the track else return the track_id and split
            yield beam.pvalue.TaggedOutput(split, track_id)



class DagstuhlToTfExample(beam.DoFn):
    DOWNLOAD_ATTRIBUTES = ["audio_lrx_path", "score_path", "f0_crepe_lrx_path" ]

    def __init__(self, source: str):
        self.source = source

    def setup(self):
        import apache_beam as beam
        import mirdata

        self.dagstuhl_remote = mirdata.initialize("dagstuhl_choirset", data_home=os.path.join(self.source, DAGSTUHL_DIR))
        self.filesystem = beam.io.filesystems.FileSystems()

    def process(self, element: List[str]):
        import tempfile

        import mirdata
        import numpy as np
        import sox
        import soundfile as sf

        from basic_pitch.constants import (
            AUDIO_N_CHANNELS,
            AUDIO_SAMPLE_RATE,
            FREQ_BINS_CONTOURS,
            FREQ_BINS_NOTES,
            ANNOTATION_HOP,
            N_FREQ_BINS_NOTES,
            N_FREQ_BINS_CONTOURS,
        )
        from basic_pitch.dataset import tf_example_serialization

        logging.info(f"Processing {element}")
        batch = []

        for track_id in element:
            track_remote = self.dagstuhl_remote.track(track_id)
            with tempfile.TemporaryDirectory() as local_tmp_dir:
                dagstuhl_local = mirdata.initialize("dagstuhl_choirset", local_tmp_dir)
                track_local = dagstuhl_local.track(track_id)

                for attribute in self.DOWNLOAD_ATTRIBUTES:
                    source = getattr(track_remote, attribute)
                    destination = getattr(track_local, attribute)
                    os.makedirs(os.path.dirname(destination), exist_ok=True)
                    with self.filesystem.open(source) as s, open(destination, "wb") as d:
                        d.write(s.read())

                local_wav_path = f"{track_local.audio_lrx_path}_tmp.wav"

                tfm = sox.Transformer()
                tfm.rate(AUDIO_SAMPLE_RATE)
                tfm.channels(AUDIO_N_CHANNELS)
                tfm.build(track_local.audio_lrx_path, local_wav_path)                
                data, samplerate = sf.read(local_wav_path)    
                sf.write(local_wav_path, data, samplerate, subtype='PCM_16')


                duration = sox.file_info.duration(local_wav_path)
                time_scale = np.arange(0, duration + ANNOTATION_HOP, ANNOTATION_HOP)
                n_time_frames = len(time_scale)

                note_indices, note_values = track_local.score.to_sparse_index(
                    time_scale, "s", FREQ_BINS_NOTES, "hz"
                )
                onset_indices, onset_values = track_local.score.to_sparse_index(
                    time_scale, "s", FREQ_BINS_NOTES, "hz", onsets_only=True
                )
                contour_indices, contour_values = track_local.f0_crepe_lrx.to_sparse_index(
                    time_scale, "s", FREQ_BINS_CONTOURS, "hz"
                )

                batch.append(
                    tf_example_serialization.to_transcription_tfexample(
                        track_local.track_id,
                        "dagstuhl_choirset",
                        local_wav_path,
                        note_indices,
                        note_values,
                        onset_indices,
                        onset_values,
                        contour_indices,
                        contour_values,
                        (n_time_frames, N_FREQ_BINS_NOTES), # shape of note_indices and onset_indices
                        (n_time_frames, N_FREQ_BINS_CONTOURS), # shape of contour_indices
                    )
                )
        return [batch]


def create_input_data(
    train_percent: float, validation_percent: float, seed: Optional[int] = None
) -> List[Tuple[str, str]]:
    assert train_percent + validation_percent < 1.0, "Don't over allocate the data!"

    # Test percent is 1 - train - validation
    validation_bound = train_percent
    test_bound = validation_bound + validation_percent

    if seed:
        random.seed(seed)

    def determine_split() -> str:
        partition = random.uniform(0, 1)
        if partition < validation_bound:
            return "train"
        if partition < test_bound:
            return "validation"
        return "test"

    dagstuhl = mirdata.initialize("dagstuhl_choirset")
    #dagstuhl.download()

    return [(track_id, determine_split()) for track_id in dagstuhl.track_ids]


def main(known_args, pipeline_args):
    time_created = int(time.time())
    destination = commandline.resolve_destination(known_args, time_created)
    input_data = create_input_data(known_args.train_percent, known_args.validation_percent, known_args.split_seed)

    pipeline_options = {
        "runner": known_args.runner,
        "job_name": f"dagstuhl-tfrecords-{time_created}",
        "machine_type": "e2-standard-4",
        "num_workers": 25,
        "disk_size_gb": 128,
        "experiments": ["use_runner_v2"],
        "save_main_session": True,
        "worker_harness_container_image": known_args.worker_harness_container_image,
    }
    pipeline.run(
        pipeline_options,
        input_data,
        DagstuhlToTfExample(known_args.source),
        DagstuhlInvalidTracks(known_args.source),
        destination,
        known_args.batch_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commandline.add_default(parser, op.basename(op.splitext(__file__)[0]))
    commandline.add_split(parser)
    known_args, pipeline_args = parser.parse_known_args()  # parser.parse_known_args(sys.argv)

    main(known_args, pipeline_args)
