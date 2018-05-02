#!/usr/bin/env python3
"""
Konwersja wszystkich plikow z katalogu mp3 do formatu wav.
"""

from pydub import AudioSegment
from os import listdir
from os.path import isfile, join

files = [f for f in listdir('mp3') if isfile(join('mp3/', f))]

for file in files:
    print('Converting %s' % file)
    sound = AudioSegment.from_mp3("mp3/%s" % file)
    sound.export("wav/%s.wav" % file.split('.')[0], format="wav")
