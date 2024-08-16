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
logging.basicConfig(level=logging.INFO)
import os
import os.path as op
import random
import time
from typing import List, Tuple, Optional
import pandas as pd

import apache_beam as beam
import pretty_midi


from basic_pitch.dataset import commandline, pipeline

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pretty_midi")

HWD_DIR = "hwd"  # "HumsandWhistlesDataset"
# dictionary with start time, instruments to keep and initial tempo for each song
HWD_DICT = {
        'Frozen': (67, [0], 120),
        'Hakuna': (54, [6,7], 180),
        'StarWars': (8, [0], 112),
        'Panther': (23.5, [5], 120),
        'Mamma': (48, [0], 140),
        'Potter': (0, [0], 90),
        'Rain': (8, [0], 120),
        'Showman': (44.5, [0], 90)
    }

def crop_and_shift_midi(midi_path, midi_output, duration, song):
    # load MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    # get start time and instruments to keep
    start_time, tokeep_instruments, initial_tempo = HWD_DICT[song]    
    # compute end time
    end_time = start_time + duration
    # new midi object to store the cropped and modified MIDI

    cropped_midi = pretty_midi.PrettyMIDI(initial_tempo=initial_tempo)
    
    for idx, instrument in enumerate(midi_data.instruments):
        if idx in tokeep_instruments:
        # new instrument in the cropped MIDI
            new_instrument = pretty_midi.Instrument(program=instrument.program)
            
            for note in instrument.notes:
                if note.start >= start_time and note.end <= end_time:

                    # No more sax notes after 15 seconds in Hakuna
                    if song == 'Hakuna' and idx == 7 and note.start > start_time+15:
                        continue

                    # Shift temporale delle note
                    new_start = note.start - start_time
                    new_end = note.end - start_time
                    # Crea una nuova nota con i tempi shiftati
                    new_note = pretty_midi.Note(
                        pitch=note.pitch,
                        start=new_start,
                        end=new_end,
                        velocity=note.velocity
                    )
                    new_instrument.notes.append(new_note)
                
            # Aggiungi lo strumento al nuovo file MIDI
            cropped_midi.instruments.append(new_instrument)

    # Salva il nuovo file MIDI
    cropped_midi.write(midi_output)
    return

import numpy as np 
import pretty_midi
import librosa
from basic_pitch.constants import (
    ANNOTATION_HOP,
    ANNOTATIONS_BASE_FREQUENCY,
    CONTOURS_BINS_PER_SEMITONE,
    NOTES_BINS_PER_SEMITONE,
)


def create_onset(midi_path):

    # Load MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    onsets_indices = np.empty((0, 2))

    # Get onsets indices
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            row = [note.start, note.pitch]
            onsets_indices = np.vstack((onsets_indices, row))
    # Translate from time to frame index
    onsets_indices[:,0] = np.round(onsets_indices[:,0] / ANNOTATION_HOP)

    # Translate from MIDI pitch to frequency bin
    onsets_indices[:,1] = librosa.midi_to_hz(onsets_indices[:,1])
    onsets_indices[:,1] = 12.0 * NOTES_BINS_PER_SEMITONE * np.log2(onsets_indices[:,1] / ANNOTATIONS_BASE_FREQUENCY)
    onsets_indices[:,1] = onsets_indices[:,1]

    # Round to the nearest integer (they are indices)
    onsets_indices = onsets_indices.astype(int)
    # Create onset values
    onset_values = np.ones(onsets_indices.shape[0])
    return onsets_indices, onset_values

def create_notes(midi_path):

    # Load MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    note_indices = np.empty((0, 2))

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start = np.round(note.start / ANNOTATION_HOP)
            end = np.round(note.end / ANNOTATION_HOP)
            # Duration in frames
            duration = int(end - start)
            # Translate from MIDI pitch to frequency bin
            note_bin = librosa.midi_to_hz(note.pitch)
            note_bin = 12.0 * NOTES_BINS_PER_SEMITONE * np.log2(note_bin / ANNOTATIONS_BASE_FREQUENCY)

            for i in range(duration):
                row = [start + i, note_bin]
                note_indices = np.vstack((note_indices, row))

    # Round to the nearest integer (they are indices)
    note_indices = note_indices.astype(int)

    # Create note values
    note_values = np.ones(note_indices.shape[0])
    return note_indices, note_values

def create_contour(midi_path):
    # Load MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    note_indices = np.empty((0, 2))

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start = np.round(note.start / ANNOTATION_HOP)
            end = np.round(note.end / ANNOTATION_HOP)
            # Duration in frames
            duration = int(end - start)
            # Translate from MIDI pitch to frequency bin
            note_bin = librosa.midi_to_hz(note.pitch)
            note_bin = 12.0 * CONTOURS_BINS_PER_SEMITONE * np.log2(note_bin / ANNOTATIONS_BASE_FREQUENCY)

            for i in range(duration):
                row = [start + i, note_bin]
                note_indices = np.vstack((note_indices, row))

    # Round to the nearest integer (they are indices)
    note_indices = note_indices.astype(int)

    # Create contour values
    note_values = np.ones(note_indices.shape[0])
    return note_indices, note_values

class HWDFilterInvalidTracks(beam.DoFn):
    def process(self, element: Tuple[str, str]):
        track_id, split = element
        if track_id == "StarWars_Whistle_122_1408" or track_id == "Panther_Hum_85_1940":
            return None
        yield beam.pvalue.TaggedOutput(split, track_id)


class HWDSetToTfExample(beam.DoFn):
    DOWNLOAD_ATTRIBUTES = ["audio_path", "midi_path"]

    def __init__(self, source: str):
        self.source = source

    def setup(self):
        import apache_beam as beam

        self.filesystem = beam.io.filesystems.FileSystems()

    def process(self, element: List[str]):
        import tempfile

        import numpy as np
        import sox

        from basic_pitch.constants import (
            AUDIO_N_CHANNELS,
            AUDIO_SAMPLE_RATE,
            ANNOTATION_HOP,
            N_FREQ_BINS_NOTES,
            N_FREQ_BINS_CONTOURS,
        )
        from basic_pitch.dataset import tf_example_serialization

        logging.info(f"Processing {element}")
        batch = []

        for track_id in element:
            with tempfile.TemporaryDirectory() as local_tmp_dir:

                for attr in self.DOWNLOAD_ATTRIBUTES:
                    if attr == "audio_path":
                        attr_path = os.path.join(HWD_DIR,f'MLEndHWD_{track_id[:track_id.find("_")]}_Audio_Files', f'{track_id[-4:]}.wav')
                        audio_path = attr_path
                    if attr == "midi_path":
                        attr_path = os.path.join(HWD_DIR, 'MIDI', f'{track_id[:track_id.find("_")]}.mid')
                        midi_path = attr_path
                    source = os.path.join(self.source, attr_path)
                    dest = os.path.join(local_tmp_dir, attr_path)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with self.filesystem.open(source) as s, open(dest, "wb") as d:
                        d.write(s.read())

                local_wav_path = "{}_tmp.wav".format(os.path.join(local_tmp_dir, audio_path))
                tfm = sox.Transformer()
                tfm.rate(AUDIO_SAMPLE_RATE)
                tfm.channels(AUDIO_N_CHANNELS)
                tfm.build(os.path.join(local_tmp_dir, audio_path), local_wav_path)

                
                duration = sox.file_info.duration(local_wav_path)
                time_scale = np.arange(0, duration + ANNOTATION_HOP, ANNOTATION_HOP)
                n_time_frames = len(time_scale)

                crop_and_shift_midi(midi_path=os.path.join(local_tmp_dir,midi_path), midi_output=os.path.join(local_tmp_dir, f'{track_id}.mid'), duration=duration, song=track_id[:track_id.find("_")])

                note_indices, note_values = create_notes(midi_path= os.path.join(local_tmp_dir, f'{track_id}.mid'))
                onset_indices, onset_values = create_onset(midi_path= os.path.join(local_tmp_dir, f'{track_id}.mid'))
                contour_indices, contour_values = create_contour(midi_path= os.path.join(local_tmp_dir, f'{track_id}.mid'))
                                
                batch.append(
                    tf_example_serialization.to_transcription_tfexample(
                        track_id,
                        "hwd",
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
    train_percent: float, validation_percent: float, source: str, seed: Optional[int] = None
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

    # Create a list of track_ids and their corresponding split
    track_ids = [] 
    df = pd.read_csv(os.path.join(source, 'hwd/MLEndHWD_Audio_Attributes.csv'))
    for i in range(len(df)):
        track_ids.append(f'{df.loc[i,"Song"]}_{df.loc[i,"Interpretation"]}_{df.loc[i,"Interpreter"]}_{df.loc[i,"Public filename"].removesuffix(".wav")}')

    return [(track_id, determine_split()) for track_id in track_ids]


def main(known_args, pipeline_args):
    print(known_args.source)
    time_created = int(time.time())
    destination = commandline.resolve_destination(known_args, time_created)
    input_data = create_input_data(known_args.train_percent, known_args.validation_percent, known_args.source, known_args.split_seed)

    pipeline_options = {
        "runner": known_args.runner,
        "job_name": f"hwd-tfrecords-{time_created}",
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
        HWDSetToTfExample(known_args.source),
        HWDFilterInvalidTracks(known_args.source),
        destination,
        known_args.batch_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commandline.add_default(parser, op.basename(op.splitext(__file__)[0]))
    commandline.add_split(parser)
    known_args, pipeline_args = parser.parse_known_args()  # parser.parse_known_args(sys.argv)

    main(known_args, pipeline_args)
