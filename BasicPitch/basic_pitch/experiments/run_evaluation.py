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


from basic_pitch import ICASSP_2022_MODEL_PATH, note_creation

from basic_pitch.experiments.predict import run_inference
from basic_pitch.experiments.evaluation_data import evaluation_data_generator

logger = logging.getLogger("mirdata")
logger.setLevel(logging.ERROR)


def model_inference(audio_path, model, minimum_note_length=127.70):

    output = run_inference(audio_path, model)

    # load the output from the model (matrices (n_frames, n_bins))
    frames = output["note"]
    onsets = output["onset"]

    min_note_len = int(np.round(minimum_note_length / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP))) # add min_note len since it is required

    # list of tuples [(start_time_frames, end_time_frames, pitch_midi, amplitude)]
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
    # load the pitch_midis
    pitch = np.array([n[2] for n in estimated_notes]) 
    
    # convert the pitch_midis to hz
    pitch_hz = librosa.midi_to_hz(pitch)
    
    # add to the estimated notes the pitch_bends lists
    estimated_notes_with_pitch_bend = note_creation.get_pitch_bends(output["contour"],estimated_notes)
    # convert the frames to real time
    times_s = note_creation.model_frames_to_time(output["contour"].shape[0])
    
    estimated_notes_time_seconds = [
        (times_s[note[0]], times_s[note[1]], note[2], note[3], note[4]) for note in estimated_notes_with_pitch_bend
    ]

    # create the midi file from the estimated notes
    midi = note_creation.note_events_to_midi(estimated_notes_time_seconds)

    # get the intervals of the notes
    intervals = np.array([[times_s[note[0]], times_s[note[1]]] for note in estimated_notes_with_pitch_bend])
    

    return intervals, pitch_hz, midi 



def main(model_name: str, data_home: str) -> None:

    #Load the trained model
    model_path = "{}".format(model_name)
    model = tf.saved_model.load(model_path)
    # Print the weights of the model
    for variable in model.variables:
        print(variable.name, variable.shape)

    save_dir = os.path.join("model_outputs", model_name)
    os.makedirs(save_dir, exist_ok=True)
    # create the save directory if it does not exist
    all_track_generator = evaluation_data_generator(data_home)
    scores = {}
    
    # iterate trough a generator that yields the dataset, track_id, instrument, audio_path, and note_data
    for dataset, track_id, instrument, audio_path, note_data in all_track_generator:
        print("[{}] {}: {}".format(dataset, track_id, instrument))
        
        # create the save path for the midi file
        save_path = os.path.join(save_dir, "{}.mid".format(track_id.replace("/", "-")))

        if os.path.exists(save_path):
            est_notes = io.load_notes_from_midi(save_path)
            if est_notes is None:
                est_intervals = []
                est_pitches = []
            else:
                # estimated intervals of times (start and end) and pitches from the midi file
                est_intervals, est_pitches, _ = est_notes.to_mir_eval()
        
        # if the midi file does not exist, run the model inference
        else:
            __,_,midi = model_inference(audio_path, model)

            est_notes = io.load_notes_from_midi(midi = midi)
            if est_notes is None:
                est_intervals = []
                est_pitches = []
            else:
                est_intervals, est_pitches, _ = est_notes.to_mir_eval()

        # get the reference intervals and pitches from the annotations
        if dataset == 'hwd':
            ref_intervals = note_data['onset']
            ref_pitches = note_data['pitch']
        else:        
            ref_intervals, ref_pitches, _ = note_data.to_mir_eval()

        if len(est_intervals) == 0 or len(ref_intervals) == 0:
            scores_trackid = {}
        else:
            # evaluate the model output obtaining the scores as a dictionary
            scores_trackid = mir_eval.transcription.evaluate(ref_intervals, ref_pitches, est_intervals, est_pitches)

        # add the scores to the scores dictionary
        scores[track_id] = scores_trackid
        scores[track_id]["instrument"] = instrument

    
    # save the scores to a json file
    with open(os.path.join(save_dir,f'{model_name}.json'), "w") as fhandle:
        json.dump(scores, fhandle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-m",
        "--model",
        default=os.path.join(os.path.expanduser('~'), "/home/seraf/DLAI/Advanced_Pitch/BasicPitch/My_models/model.best/"),
        type=str,
        help="Path to the saved model directory.",
    )
    parser.add_argument("--data-home", default=os.path.join(os.path.expanduser('~'),'mir_datasets'), type=str, help="Location to store evaluation data.")
    args = parser.parse_args()

    main(args.model, args.data_home)
