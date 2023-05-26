import presets as pre
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
import bisect


def get_notes():
	notes = [(freq * (2 ** (octave - pre.PRELIM_NOTES_OCTAVE)), f"{name}{octave}") for octave in range(pre.OCTAVES) for name, freq in pre.PRELIM_NOTES]
	return notes


def get_notes_dict():
	return {name: freq for freq, name in get_notes()}


def load_fourier_tensor(n):
	notes = get_notes()
	fourier_row = torch.arange(n).type(torch.float64)
	fourier_row = (-2 * np.pi * fourier_row) / pre.SAMPLE_RATE

	fourier_matrix = fourier_row.repeat(len(notes), 1)
	for row, (frequency, _) in zip(fourier_matrix, notes):
		row *= frequency
	return torch.cos(fourier_matrix), torch.sin(fourier_matrix)


def get_fourier_transform(wavelet, fourier_tensors=None):
	if fourier_tensors is None:
		fourier_matrix_cos, fourier_matrix_sin = load_fourier_tensor(len(wavelet))
	else:
		fourier_matrix_cos, fourier_matrix_sin = fourier_tensors
	return [torch.tensordot(fourier_matrix, wavelet, dims=1) for fourier_matrix in (fourier_matrix_cos, fourier_matrix_sin)]


def get_amplitude_spectrum(wavelet, fourier_tensors=None):
	if type(wavelet) != torch.Tensor or wavelet.dtype != torch.float64:
		wavelet = torch.Tensor(wavelet).type(torch.float64)
	(f_hat_cos, f_hat_sin) = get_fourier_transform(wavelet, fourier_tensors=fourier_tensors)
	frequencies = [note[0] for note in get_notes()]
	amplitude_spectrum = np.linalg.norm(np.stack((f_hat_cos.numpy(), f_hat_sin.numpy()), axis=1), axis=1)
	return frequencies, amplitude_spectrum


def load_frequency_matrix():
	notes = get_notes()
	matrix = np.identity(len(notes))
	for row_index in range(len(matrix) - pre.NOTES_IN_OCTAVE):
		matrix[row_index][row_index+pre.NOTES_IN_OCTAVE] = pre.OCTAVE_RATIO
	return matrix


def lsq_frequencies(wavelet, fourier_tensors=None, freq_tensor=None):
	if freq_tensor is None:
		freq_tensor = load_frequency_matrix()
	frequencies, amplitude_spectrum = get_amplitude_spectrum(wavelet, fourier_tensors=fourier_tensors)
	return np.linalg.lstsq(freq_tensor, amplitude_spectrum, rcond=None)[0]


def gaussian_kernel(length, std=1):
    assert length % 2
    center = length // 2
    kernel = []
    for i in range(int(length)):
        distance = abs(i - center)
        kernel.append(np.exp(-(distance * distance) / (2 * std * std)))
    kernel = np.array(kernel)
    kernel = kernel - (np.sum(kernel) / length)
    kernel = kernel / np.linalg.norm(kernel)
    return kernel


def parse_amplitudes(note_ls, amplitude_ls):
	parsed_notes = []

	combined = [(note, amplitude) for note, amplitude in zip(note_ls, amplitude_ls)]
	combined.sort()
	for i, (note, amplitude) in enumerate(combined):
		if i and combined[i-1][0][:-1] == note[:-1] and int(combined[i-1][0][-1]) == int(note[-1]) - 1 and combined[i-1][1] - amplitude > pre.max_diff_amplitude:
			continue
		parsed_notes.append(note)
	return parsed_notes


def predict_freqs(wavelet, fourier_tensors=None, freq_tensor=None, gkernel=None):
	if gkernel is None:
		gkernel = gaussian_kernel(pre.gkernel_length, pre.gkernel_std)
	amplitudes = lsq_frequencies(wavelet, fourier_tensors=fourier_tensors, freq_tensor=freq_tensor)
	conv = np.convolve(amplitudes, gkernel, mode='same')
    
	amplitude_threshold = np.mean(amplitudes) + (pre.zscore_thresh * np.std(amplitudes))
	predicted_frequencies = (amplitudes > amplitude_threshold) * (conv > pre.normal_threshold)
	note_ls = list(np.array([name for _, name in get_notes()])[predicted_frequencies])
	amplitude_ls = amplitudes[predicted_frequencies]
	return parse_amplitudes(note_ls, amplitude_ls)


def get_kmers(wav):
    return (wav[i:i+pre.K] for i in range(0, len(wav), pre.STEP) if i+pre.K <= len(wav))


def transcribe_wavelet(wavelet, log=False):
	fourier_tensors = load_fourier_tensor(pre.K)
	freq_tensor = load_frequency_matrix()
	gkernel = gaussian_kernel(pre.gkernel_length, pre.gkernel_std)

	note_list = []
	for i, kmer in enumerate(get_kmers(wavelet)):
		note_ls = predict_freqs(kmer, fourier_tensors=fourier_tensors, freq_tensor=freq_tensor, gkernel=gkernel)
		if log and not (i * pre.STEP  // pre.SAMPLE_RATE) * 100 % 10:
			print(f"Start: second {i * pre.STEP  // pre.SAMPLE_RATE}", note_ls)
		note_list.append(note_ls)
	return note_list


def format_note_list(note_list):
	notes_durations = []
	playing = {}
	for i, note_ls in enumerate(note_list):
		for note in note_ls:
			if note not in playing:
				playing[note] = i
		remove = []
		for note in playing:
			if note not in note_ls:
				notes_durations.append((note, playing[note], i))
				remove.append(note)
		for note in remove:
			del playing[note]
	for note in playing:
		notes_durations.append((note, playing[note], i))
	return notes_durations


def get_frequency(frequency, duration=1):
	dur_base = np.arange(0, pre.K * duration).astype(np.float64)
    
	c = (frequency * 2 * np.pi) / pre.SAMPLE_RATE
	wavelet_ = pre.RECONSTRUCTION_AMPLITUDE * np.sin(c * dur_base)
	return wavelet_


def reconstruct(note_list):
	base = np.arange(0, pre.K).astype(np.float64)
	wav = np.tile(np.zeros_like(base), len(note_list))
	note_durations = format_note_list(note_list)
    
	notes_dict = get_notes_dict()
	for note, start, end in note_durations:
		wav[start*len(base):end*len(base)] += get_frequency(notes_dict[note], end - start)
	return wav, pre.SAMPLE_RATE
