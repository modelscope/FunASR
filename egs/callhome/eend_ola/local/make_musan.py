#!/usr/bin/env python3
# Copyright 2015   David Snyder
#           2018   Ewald Enzinger
# Apache 2.0.
#
# Modified version of egs/sre16/v1/local/make_musan.py (commit e3fb7c4a0da4167f8c94b80f4d3cc5ab4d0e22e8).
# This version uses the raw MUSAN audio files (16 kHz) and does not use sox to resample at 8 kHz.
#
# This file is meant to be invoked by make_musan.sh.

import os, sys

def process_music_annotations(path):
  utt2spk = {}
  utt2vocals = {}
  lines = open(path, 'r').readlines()
  for line in lines:
    utt, genres, vocals, musician = line.rstrip().split()[:4]
    # For this application, the musican ID isn't important
    utt2spk[utt] = utt
    utt2vocals[utt] = vocals == "Y"
  return utt2spk, utt2vocals

def prepare_music(root_dir, use_vocals):
  utt2vocals = {}
  utt2spk = {}
  utt2wav = {}
  num_good_files = 0
  num_bad_files = 0
  music_dir = os.path.join(root_dir, "music")
  for root, dirs, files in os.walk(music_dir):
    for file in files:
      file_path = os.path.join(root, file)
      if file.endswith(".wav"):
        utt = str(file).replace(".wav", "")
        utt2wav[utt] = file_path
      elif str(file) == "ANNOTATIONS":
        utt2spk_part, utt2vocals_part = process_music_annotations(file_path)
        utt2spk.update(utt2spk_part)
        utt2vocals.update(utt2vocals_part)
  utt2spk_str = ""
  utt2wav_str = ""
  for utt in utt2vocals:
    if utt in utt2wav:
      if use_vocals or not utt2vocals[utt]:
        utt2spk_str = utt2spk_str + utt + " " + utt2spk[utt] + "\n"
        utt2wav_str = utt2wav_str + utt + " " + utt2wav[utt] + "\n"
      num_good_files += 1
    else:
      print("Missing file {}".format(utt))
      num_bad_files += 1
  print("In music directory, processed {} files: {} had missing wav data".format(num_good_files, num_bad_files))
  return utt2spk_str, utt2wav_str

def prepare_speech(root_dir):
  utt2spk = {}
  utt2wav = {}
  num_good_files = 0
  num_bad_files = 0
  speech_dir = os.path.join(root_dir, "speech")
  for root, dirs, files in os.walk(speech_dir):
    for file in files:
      file_path = os.path.join(root, file)
      if file.endswith(".wav"):
        utt = str(file).replace(".wav", "")
        utt2wav[utt] = file_path
        utt2spk[utt] = utt
  utt2spk_str = ""
  utt2wav_str = ""
  for utt in utt2spk:
    if utt in utt2wav:
      utt2spk_str = utt2spk_str + utt + " " + utt2spk[utt] + "\n"
      utt2wav_str = utt2wav_str + utt + " " + utt2wav[utt] + "\n"
      num_good_files += 1
    else:
      print("Missing file {}".format(utt))
      num_bad_files += 1
  print("In speech directory, processed {} files: {} had missing wav data".format(num_good_files, num_bad_files))
  return utt2spk_str, utt2wav_str

def prepare_noise(root_dir):
  utt2spk = {}
  utt2wav = {}
  num_good_files = 0
  num_bad_files = 0
  noise_dir = os.path.join(root_dir, "noise")
  for root, dirs, files in os.walk(noise_dir):
    for file in files:
      file_path = os.path.join(root, file)
      if file.endswith(".wav"):
        utt = str(file).replace(".wav", "")
        utt2wav[utt] = file_path
        utt2spk[utt] = utt
  utt2spk_str = ""
  utt2wav_str = ""
  for utt in utt2spk:
    if utt in utt2wav:
      utt2spk_str = utt2spk_str + utt + " " + utt2spk[utt] + "\n"
      utt2wav_str = utt2wav_str + utt + " " + utt2wav[utt] + "\n"
      num_good_files += 1
    else:
      print("Missing file {}".format(utt))
      num_bad_files += 1
  print("In noise directory, processed {} files: {} had missing wav data".format(num_good_files, num_bad_files))
  return utt2spk_str, utt2wav_str

def main():
  in_dir = sys.argv[1]
  out_dir = sys.argv[2]
  use_vocals = sys.argv[3] == "Y"
  utt2spk_music, utt2wav_music = prepare_music(in_dir, use_vocals)
  utt2spk_speech, utt2wav_speech = prepare_speech(in_dir)
  utt2spk_noise, utt2wav_noise = prepare_noise(in_dir)
  utt2spk = utt2spk_speech + utt2spk_music + utt2spk_noise
  utt2wav = utt2wav_speech + utt2wav_music + utt2wav_noise
  wav_fi = open(os.path.join(out_dir, "wav.scp"), 'w')
  wav_fi.write(utt2wav)
  utt2spk_fi = open(os.path.join(out_dir, "utt2spk"), 'w')
  utt2spk_fi.write(utt2spk)


if __name__=="__main__":
  main()
