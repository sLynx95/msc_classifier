#!/usr/bin/env python3

"""Przetwarzanie wstępne utworu muzycznego"""

import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft
from matplotlib import pyplot as plt

# Podajemy sciezke do pliku wav
filename = 'wav/Three_Days_Grace-The_Mountain.wav'

# Wczytujemy plik wav
sample_rate, stereo_sound = wavfile.read(filename)

print(sample_rate)

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

# Przekształcamy audio na reprezentację z oknami w wierszach i próbkami
# w kolumnach
windows = mono_sound.reshape((n_windows, -1))
# print(np.shape(windows))

# Iterujemy kolejne okna
for i in range(n_windows):
    # Pobieramy okno
    window = windows[i, :]

    # Dla przykładu wyliczamy i wyświetlamy średnią z okna
    mean_of_window = np.mean(window)
    # print(mean_of_window)

    # Wyliczamy szybka transformate fouriera dla okna
    fft_from_window = fft(window)

    # Z FFT wyliczmy czestotliwości
    frequencies = np.abs(fft_from_window)

    # Wyliczamy szerokosc pasma dla okna
    bandwidth_of_window = np.max(frequencies) - np.min(frequencies)
    # print(bandwidth_of_window)

    # Dla pierwszego i drugiego okna rysujemy wykres przebiegu w czasie oraz częstotliwości
    if i == 0 or i == 1:
        fig = plt.figure(dpi=128, figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(window)
        plt.subplot(2, 1, 2)
        plt.plot(frequencies)
        plt.show()
