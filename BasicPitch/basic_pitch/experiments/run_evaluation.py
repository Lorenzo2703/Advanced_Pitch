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
import json
import logging
import os

from basic_pitch.constants import AUDIO_SAMPLE_RATE, FFT_HOP
import librosa
import mir_eval
import numpy as np
import tensorflow as tf
from mirdata import io

from basic_pitch import note_creation

from predict import run_inference
from evaluation_data import evaluation_data_generator

logger = logging.getLogger("mirdata")
logger.setLevel(logging.ERROR)


def model_inference(audio_path, model, save_path,minimum_note_length=127.70):

    output = run_inference(audio_path, model)
    frames = output["note"]
    onsets = output["onset"]

    min_note_len = int(np.round(minimum_note_length / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP))) # add min_note len since it is required


    estimated_notes = note_creation.output_to_notes_polyphonic(
        frames,
        onsets,
        onset_thresh=0.5,
        frame_thresh=0.3,
        infer_onsets=True,
        min_note_len=min_note_len, # needed in the function, it will throw error if not provided
        max_freq=None, # needed in the function, it will throw error if not provided
        min_freq=None # needed in the function, it will throw error if not provided
    )
    # [(start_time_seconds, end_time_seconds, pitch_midi, amplitude)]
    pitch = np.array([n[2] for n in estimated_notes]) 
    pitch_hz = librosa.midi_to_hz(pitch)
    estimated_notes_with_pitch_bend = note_creation.get_pitch_bends(output["contour"],estimated_notes)
    times_s = note_creation.model_frames_to_time(output["contour"].shape[0])
    
    estimated_notes_time_seconds = [
        (times_s[note[0]], times_s[note[1]], note[2], note[3], note[4]) for note in estimated_notes_with_pitch_bend
    ]

    midi = note_creation.note_events_to_midi(estimated_notes_time_seconds, save_path)

    intervals = np.array([[times_s[note[0]], times_s[note[1]]] for note in estimated_notes_with_pitch_bend])
    

    return intervals, pitch_hz,midi # add midi in the return to be used in the evaluation

    
    # intervals = np.array([[n[0], n[1]] for n in estimated_notes])
    # pitch_hz = librosa.midi_to_hz(np.array([n[2] for n in estimated_notes]))

    # note_creation.note_events_to_midi(estimated_notes, save_path)

    # return intervals, pitch_hz


def main(model_name: str, data_home: str) -> None:
    model_path = "{}".format(model_name)
    model = tf.saved_model.load(model_path)

    save_dir = os.path.join("model_outputs", model_name)

    all_track_generator = evaluation_data_generator(data_home)
    scores = {}
    for dataset, track_id, instrument, audio_path, note_data in all_track_generator:
        print("[{}] {}: {}".format(dataset, track_id, instrument))
        save_path = os.path.join(save_dir, "{}.mid".format(track_id.replace("/", "-")))

        if os.path.exists(save_path):
            est_notes = io.load_notes_from_midi(save_path)
            if est_notes is None:
                est_intervals = []
                est_pitches = []
            else:
                est_intervals, est_pitches, _ = est_notes.to_mir_eval()
        else:
            # est_intervals, est_pitches = model_inference(audio_path, model, save_path)
            __,_,midi = model_inference(audio_path, model, save_path)

            est_notes = io.load_notes_from_midi(midi = midi)
            if est_notes is None:
                est_intervals = []
                est_pitches = []
            else:
                est_intervals, est_pitches, _ = est_notes.to_mir_eval()
                
        ref_intervals, ref_pitches, _ = note_data.to_mir_eval()

        if len(est_intervals) == 0 or len(ref_intervals) == 0:
            scores_trackid = {}
        else:
            scores_trackid = mir_eval.transcription.evaluate(ref_intervals, ref_pitches, est_intervals, est_pitches)

        scores[track_id] = scores_trackid
        scores[track_id]["instrument"] = instrument

    with open("{}.json".format(model_name), "w") as fhandle:
        json.dump(scores, fhandle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Which model to run evaluation on",
    )
    parser.add_argument("--data-home", type=str, help="Location to store evaluation data.")
    args = parser.parse_args()

    main(args.model, args.data_home)
