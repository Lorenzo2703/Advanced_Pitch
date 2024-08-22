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

import csv
from itertools import chain
import os
import mirdata
import numpy as np

import sox
from basic_pitch.dataset import hwd


def load_test_tracks(dataset):
    # get the path to experiments folder
    currdir = os.path.dirname(os.path.realpath(__file__))
    
    # get the path to the test_tracks.csv file of that dataset
    fname = "{}_test_tracks.csv".format(dataset)

    # read the track ids from the file
    with open(os.path.join(currdir, "split_ids", fname), "r") as fhandle:
        reader = csv.reader(fhandle)
        track_ids = [line[0] for line in reader]
    return track_ids


def update_data_home(data_home, dataset_name):

    if data_home is None:
        return data_home

    if data_home.startswith("gs://"):
        return data_home

    # load the path to the  specific dataset
    if dataset_name not in data_home:
        return os.path.join(data_home, dataset_name)

def hwd_tracks(data_home):
    test_tracks = load_test_tracks("hwd")

    for track_id in test_tracks:
        instrument = track_id.split("_")[1].lower()
        audio_path = os.path.join(os.path.expanduser('~'), 'mir_datasets', 'hwd', f'MLEndHWD_{track_id.split("_")[0]}_Audio_Files' ,f'{track_id.split("_")[-1]}.wav')
        full_midi_path = os.path.join(os.path.expanduser('~'), 'mir_datasets', 'hwd', 'MIDI',f'{track_id.split("_")[0]}.mid')
        duration = sox.file_info.duration(audio_path)
        midi_data = hwd.crop_and_shift_midi(midi_path=full_midi_path, duration=duration, song=track_id.split("_")[0])
        onset_indices, _ = hwd.create_onset(midi_data=midi_data,evaluation=True)

        note_data = {}
        note_data['onset'] = np.column_stack((onset_indices[:,0], onset_indices[:,1]))
        note_data['pitch'] = onset_indices[:,2]
        
        yield ("hwd", track_id, instrument, audio_path, note_data)

def maestro_tracks(data_home, limit=20):
    # load the test tracks ids for maestro dataset
    test_tracks = load_test_tracks("maestro")
    # if limit is set, only take the first limit number of tracks
    if limit:
        test_tracks = test_tracks[:limit]

    # initialize the maestro dataset
    maestro = mirdata.initialize("maestro", data_home=data_home)

    # create a generator for the test tracks
    for track_id in test_tracks:
        track = maestro.track(track_id)
        yield ("maestro", track_id, "piano", track.audio_path, track.notes)


def guitarset_tracks(data_home):
    test_tracks = load_test_tracks("guitarset")

    guitarset = mirdata.initialize("guitarset", data_home=data_home)

    for track_id in test_tracks:
        track = guitarset.track(track_id)
        yield ("guitarset", track_id, "guitar", track.audio_mic_path, track.notes_all)


def slakh_tracks(data_home, limit=100):
    test_tracks = load_test_tracks("slakh")
    if limit:
        test_tracks = test_tracks[:limit]
    
    slakh = mirdata.initialize("slakh", data_home=data_home, version='baby')
    
    for track_id in test_tracks:
        track = slakh.track(track_id)
        if track.audio_path is None or track.is_drum or track.notes is None:
            continue

        yield ("slakh", track.track_id, track.instrument, track.audio_path, track.notes)



def dagstuhl_tracks_singlevoice(data_home):
    test_tracks = load_test_tracks("dagstuhl_choirset")

    dagstuhl = mirdata.initialize("dagstuhl_choirset", data_home=data_home)

    for track_id in test_tracks:
        track = dagstuhl.track(track_id)
        if track.audio_lrx_path is None or track.score is None:
            continue

        yield ("dagstuhl_choirset", track.track_id, "vocals", track.audio_lrx_path, track.score)



def evaluation_data_generator(data_home, maestro_limit=None, slakh_limit=None):
    all_track_generator = chain(
        hwd_tracks(update_data_home(data_home, "hwd")),
        guitarset_tracks(update_data_home(data_home, "guitarset")),
        slakh_tracks(update_data_home(data_home, "slakh"), limit=slakh_limit),
        # dagstuhl_tracks_singlevoice(update_data_home(data_home, "dagstuhl_choirset")),
    )
    return all_track_generator
