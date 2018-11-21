#!/usr/bin/env python3

"""Przetwarzanie wstępne i ekstrakcja cech utworów muzycznych"""

import os
from scipy.io import wavfile
from scipy.fftpack import fft
from pandas import DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Lista na połączenie utworów z cechami
FULL_DATASET = DataFrame(np.array([]))

# Katalogi
DIRS = [d for d in os.listdir('audios')]
os.chdir('audios/')
# Petla po katalogach
for DIR in DIRS:
    genre = str(DIR).title()
    FILES = [f for f in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, f))]
    for file in FILES:
        # Podajemy sciezke do pliku wav
        wave_file = (DIR + "/%s.wav" % file.split('.')[0])
        # Wczytujemy plik wav
        sample_rate, mono_sound = wavfile.read(wave_file)
        print(file.upper())
        # Pobieramy sumaryczną długość strumienia audio
        sound_length = len(mono_sound)
        # Deklarujemy szerokość okna w sekundach
        window_seconds = 3
        # Obliczamy szerokość okna w bitach
        window_length = window_seconds * sample_rate
        # Wyliczamy liczbę okien
        n_windows = sound_length // window_length
        # Przycinamy dźwięk do pełnych okien
        mono_sound = mono_sound[:n_windows * window_length]
        # Przekształcamy audio na reprezentację z oknami w wierszach i próbkami w kolumnach
        windows = mono_sound.reshape((n_windows, -1))

        # Ilość próbek utworu
        # samples = np.shape(windows[-1])
        # n_samples = samples[0]

        # Listy na cechy okien: (szerokość pasma, czestotliwość dominujaca, moc, obwiednia)
        bandwidths, pervasives_freq, signal_strengths, signal_envelopes, genres = [], [], [], [], []
        name_features = np.array(
            ['genre', 'bandwidth', 'pervasive_freq', 'signal_strength', 'signal_envelope'])
        print('Extraction features of %s' % file)
        # Iterujemy kolejne okna
        for i in range(n_windows):
            # Pobieramy okno
            window = windows[i, :]

            # Wliczamy średnią z okna
            # mean_of_window = np.mean(window)
            # Wyliczamy odchylenie standarowe dla okna
            # stdeviation = np.std(window)

            # Wyliczamy szybka transformate fouriera dla okna
            fft_from_window = np.array(fft(window))[:int(window_length/2)]
            # Potencjalne wykorzystanie filtracji medianowej/gaussowskiej

            # Z FFT wyliczmy czestotliwości dla okna
            frequencies = np.abs(fft_from_window)
            # Wyliczamy szerokosc pasma dla okna
            bandwidth = np.max(frequencies) - np.min(frequencies)
            # Wyznaczamy moc sygnału
            signal_strength = np.sum(frequencies)
            if bandwidth and signal_strength:
                bandwidths.append(bandwidth)
                signal_strengths.append(signal_strength)
            else:
                continue
            # Wyznaczamy czestotliwość dominującą
            if np.max(window):
                pervasive_freq = frequencies[i]
                pervasives_freq.append(pervasive_freq)
            # Wyznaczamy obwiednią sygnału dla okna (z 100 pod okien)
            if np.max(frequencies):
                signal_envelope = np.sum([
                    (np.abs(np.mean(window[i]) - np.mean(window[i - 1]))) / (100 - 1)
                    for i in range(1, 100)
                ])
                signal_envelopes.append(signal_envelope)
            # Wyznaczanie mediany dla trzech pierwszych okien z cechy
            if i == 3:
                med_band = np.median(bandwidths)
                med_sig_str = np.median(signal_strengths)
                med_per_fq = np.median(pervasives_freq)
                med_sig_enve = np.median(signal_envelopes)
            # Wyznaczanie median: poprzednie okna - obecne okno
            elif i > 3:
                med_band = np.median([med_band, bandwidth])
                med_sig_str = np.median([med_sig_str, signal_strength])
                med_per_fq = np.median([med_per_fq, pervasive_freq])
                med_sig_enve = np.median([med_sig_enve, signal_envelope])
        # Wektor cech dla jednego pliku audio
        features = DataFrame(np.array([
            genre, med_band, med_per_fq, med_sig_str, med_sig_enve]).reshape(1, 5), columns=name_features)
        # Zapis wektorów
        print('Creating dataset  of %s' % file)
        FULL_DATASET = FULL_DATASET.append(features)

# Zapis pełnego datasetu do .csv
# FULL_DATASET.to_csv('full_dataset_GZTAN.csv')

'''
            fig = plt.figure(dpi=128, figsize=(10, 6))
            plt.subplot(2, 1, 1)
            plt.plot(mono_sound)
            plt.subplot(2, 1, 2)
            plt.plot(frequencies)
            plt.show()
'''
