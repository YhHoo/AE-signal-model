from nptdms import TdmsFile
from dsp_tools import spectrogram_scipy, fft_scipy

tdms_file = TdmsFile('E:/Experiment_16_5_2018/Experiment 1/pos_0_1_2_3/leak/1_bar/set_1.tdms')
voltage_0 = tdms_file.object('Untitled', 'Voltage_0')
voltage_0_data = voltage_0.data
print(voltage_0_data.shape)

_, _ = fft_scipy(sampled_data=voltage_0_data, fs=1e6, visualize=True)





