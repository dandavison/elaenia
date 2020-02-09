# logss_1 looks good, and similar to librosa.display.specshow
ss = np.abs(lr.stft(y, n_fft=N_FFT))
ss2 = ss ** 2
logss = np.log(ss)
logss2 = np.log(ss2)
logss_1 = np.log(ss + 1)
logss2_1 = np.log(ss2 + 1)
logss4_1 = np.log(ss ** 4 + 1)
