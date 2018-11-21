#!/usr/bin/env python3
"""
Konwersja wszystkich plikow z katalogu mp3 do formatu wav.
"""

from pydub import AudioSegment
from os import listdir, system, chdir
from os.path import isfile, join


def convert_mp3_to_wav():

    files = [f for f in listdir('mp3') if isfile(join('mp3/', f))]
    for file in files:
        print('Converting %s' % file)
        sound = AudioSegment.from_mp3("mp3/%s" % file)
        sound.export("wav/%s.wav" % file.split('.')[0], format="wav")

def convert_au_to_wav(catalog):

    chdir(catalog)
    files = [f for f in listdir(catalog) if isfile(join(catalog, f))]
    for file in files:
        print('Converting %s' % file)
        system("sox " + str(file) + " " + str(file[:-3]) + ".wav")
        system("rename 's/k.0/k_0/g' *.wav")
    system("chmod +x *wav")
    # system("rm *.au")


convert_au_to_wav('/home/seba/dev/genres/rock')
