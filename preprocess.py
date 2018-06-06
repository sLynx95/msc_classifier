#!/usr/bin/env python3

"""Przetwarzanie wstępne i ekstrakcja cech utworów muzycznych"""

import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft
from pandas import DataFrame, read_csv, concat
from pydub import AudioSegment
from os import listdir
from os.path import isfile, join

# Lista na połączenie utworów z cechami
full_dataset = DataFrame(np.array([]))

# Ładujemy zbiór utwór - gatunek
dataset = read_csv('song_dataset.csv')
dataset_noIndex = dataset.set_index('song')
amount_songs = dataset_noIndex.size

# Tworzymy listę utworów z folderu, format mp3
files = [f for f in listdir('samplemp3') if isfile(join('samplemp3/', f))]

# Petla po wszytkich utworach
for file in files:
    # Konwersja na format wav
    sound = AudioSegment.from_mp3("samplemp3/%s" % file)
    sound.export("wav/%s.wav" % file.split('.')[0], format="wav")

    # Podajemy sciezke do pliku wav
    wave_file = ("wav/%s.wav" % file.split('.')[0])

    # Wczytujemy plik wav
    sample_rate, stereo_sound = wavfile.read(wave_file)

    # Spłaszczamy dźwięk do mono
    mono_sound = np.mean(stereo_sound, axis=1)
    
    # Pobieramy sumaryczną długość strumienia audio
    sound_length = len(mono_sound)

    # Deklarujemy szerokość okna w sekundach
    window_seconds = 4

    # Obliczamy szerokość okna w bitach
    window_length = window_seconds * sample_rate

    # Wyliczamy liczbę okien
    n_windows = sound_length // window_length

    # Przycinamy dźwięk do pełnych okien
    mono_sound = mono_sound[:n_windows * window_length]
    # print(np.shape(mono_sound))

    # Przekształcamy audio na reprezentację z oknami w wierszach i próbkamicw kolumnach
    windows = mono_sound.reshape((n_windows, -1))

    # Ilość próbek utworu
    samples = np.shape(windows[-1])
    n_samples = samples[0]

    # Listy na cechy okien: (szerokość pasma, czestotliwość dominujaca, moc, obwiednia)
    bandwidths, pervasives_freq, signal_strengths, signal_envelopes = [], [], [], []
    name_features = np.array(['bandwidth', 'pervasive_freq', 'signal_strength', 'signal_envelope'])

    # Iterujemy kolejne okna
    for i in range(n_windows):
        # Pobieramy okno
        window = windows[i, :]

        # Wliczamy średnią z okna
        mean_of_window = np.mean(window)

        # Wyliczamy odchylenie standarowe dla okna
        stdeviation = np.std(window)

        # Wyliczamy szybka transformate fouriera dla okna
        fft_from_window = np.array(fft(window))

        # Potencjalne wykorzystanie filtracji medianowej/gaussowskiej

        # Z FFT wyliczmy czestotliwości dla okna
        frequencies = np.abs(fft_from_window)

        # Wyliczamy szerokosc pasma dla okna
        bandwidth = np.max(frequencies) - np.min(frequencies)
        bandwidths.append(bandwidth)

        # Wyznaczamy czestotliwość dominującą
        if np.max(window):
            pervasive_freq = frequencies[i]
            pervasives_freq.append(pervasive_freq)

        # Wyznaczamy moc sygnału
        signal_strength = np.sum(frequencies)
        signal_strengths.append(signal_strength)

        # Wyznaczamy obwiednią sygnału dla okna (z 100 pod okien)
        signal_envelope = np.sum([
            (np.abs(np.mean(window[i]) - np.mean(window[i - 1]))) / (100 - 1)
            for i in range(1, 100)
            ])
        signal_envelopes.append(signal_envelope)

        """
        # Dla pierwszego i drugiego okna rysujemy wykres przebiegu w czasie oraz częstotliwości
        if i == 0 or i == 1:
            fig = plt.figure(dpi=128, figsize=(10, 6))
            plt.subplot(2, 1, 1)
            plt.plot(window)
            plt.subplot(2, 1, 2)
            plt.plot(frequencies)
            plt.show()
        """
    features = DataFrame(np.transpose(np.array([
        bandwidths, pervasives_freq, signal_strengths, signal_envelopes])),
        columns=name_features)
    # features.to_csv('Mountain.csv')
    if any(dataset['song'] == file):
        single_song = DataFrame(dataset_noIndex.loc[[file], ['idx', 'genre']])
        # single_song.join(features, lsuffix='_single_song', rsuffix='_features')
        song_dataset = concat([single_song, features])
    # full_dataset.append(song_dataset)

# full_dataset.to_csv('full_dataset.csv')
