#!/usr/bin/env python3

"""Przetwarzanie wstępne i ekstrakcja cech utworów muzycznych"""

from os import listdir, remove
from os.path import isfile, join
from pydub import AudioSegment
from scipy.io import wavfile
from scipy.fftpack import fft
from pandas import DataFrame, read_csv
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Lista na połączenie utworów z cechami
FULL_DATASET = DataFrame(np.array([]))

# Ładujemy zbiór utwór - gatunek i ustawiamy indeks
DATASET = read_csv('song_dataset.csv')
DATASET_IDSONG = DATASET.set_index('song')

# Tworzymy listę utworów z folderu, format mp3
FILES = [f for f in listdir('mp3') if isfile(join('mp3/', f))]

# Petla po wszytkich utworach
for file in FILES:
    # Konwersja na format wav
    print('Converting %s' % file)
    sound = AudioSegment.from_mp3("mp3/%s" % file)
    sound.export("wav/%s.wav" % file.split('.')[0], format="wav")

    # Podajemy sciezke do pliku wav
    wave_file = ("wav/%s.wav" % file.split('.')[0])

    # Wczytujemy plik wav
    sample_rate, stereo_sound = wavfile.read(wave_file)

    # Spłaszczamy dźwięk do mono jeżeli byl stereo
    print('Preprocessing %s' % file)
    if sound.channels == 2:
        mono_sound = np.mean(stereo_sound, axis=1)
    else:
        mono_sound = stereo_sound

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

    # Przekształcamy audio na reprezentację z oknami w wierszach i próbkamicw kolumnach
    windows = mono_sound.reshape((n_windows, -1))

    # Ilość próbek utworu
    samples = np.shape(windows[-1])
    n_samples = samples[0]

    # Listy na cechy okien: (szerokość pasma, czestotliwość dominujaca, moc, obwiednia)
    bandwidths, pervasives_freq, signal_strengths, signal_envelopes, genres = [], [], [], [], []
    name_features = np.array(
        ['genre', 'bandwidth', 'pervasive_freq', 'signal_strength', 'signal_envelope'])

    # Wyciągnięcie gatunku z odpowiedniego utworu
    genre = DATASET_IDSONG.get_value(file, 'genre')

    print('Extraction features of %s' % file)
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

        # Wyznaczamy moc sygnału
        signal_strength = np.sum(frequencies)

        if bandwidth and signal_strength:
            bandwidths.append(bandwidth)
            signal_strengths.append(signal_strength)
            genres.append(genre)
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
        genres, bandwidths, pervasives_freq, signal_strengths, signal_envelopes])),
                         columns=name_features)

    if any(DATASET['song'] == file):
        single_song = DataFrame({'song': [file]})

        # Połączenie utworu z cechami
        song_dataset = single_song.join(features, how='right')

        # Zapis do zmiennej każdej kolejnej kombinacji utwór-cechy
        print('Creating dataset  of %s' % file)
        FULL_DATASET = FULL_DATASET.append(song_dataset)

        # Usunięcie pliku .wave
        print('Deleting wave file')
        remove("wav/%s.wav" % file.split('.')[0])

# Zapis pełnego datasetu do .csv
FULL_DATASET.to_csv('full_dataset.csv')
