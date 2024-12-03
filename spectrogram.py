import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hamming


def plot_spectrogram(path_audio, path_output):
    sample_rate, audio_data = wavfile.read(path_audio)

    audio_duration = len(audio_data)/sample_rate

    # Step 2: Initialize ShortTimeFFT with parameters
    # Here we use a Hamming window as an example, but you can change it to any window you prefer
    nperseg = 2056  # Segment length (number of samples)
    hop = 512       # Hop interval (overlap)
    mfft = nperseg     # Oversampling factor (optional, set to same as segment length)
    scale_to = 'psd'  # Scale to power spectral density
    window = hamming(nperseg)  # Hamming window for STFT

    # Initialize the ShortTimeFFT object with the chosen parameters
    stft = ShortTimeFFT(window, hop=hop, fs=sample_rate, mfft=mfft, scale_to=scale_to)

    # Step 3: Compute the spectrogram (absolute square of the STFT)
    Sx2 = stft.spectrogram(audio_data)

    # Step 4: Plot the spectrogram
    # Convert the spectrogram to dB (log scale)
    Sx_dB = 10 * np.log10(np.fmax(Sx2, 1e-4))  # Avoid log(0) by setting a minimum value

    # Create a time vector based on the audio data length
    time_extent = stft.extent(len(audio_data))
    t_lo, t_hi = time_extent[:2]  # Time range for the plot

    # Plot the spectrogram
    plt.figure(figsize=(audio_duration,6))
    plt.imshow(Sx_dB, origin='lower', aspect='auto', extent=time_extent, cmap='magma')

    plt.xlim(t_lo, t_hi)
    plt.xticks(range(t_lo.__int__(), t_hi.__int__()), labels='')
    plt.grid(False)
    plt.tick_params(color='white', size=10, width=2)
    plt.box(on=False)
    plt.ylim(20, 5000)  # Limit the frequency range to Nyquist frequency

    plt.savefig(path_output, transparent=True, dpi=80)