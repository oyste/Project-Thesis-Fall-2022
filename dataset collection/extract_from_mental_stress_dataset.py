import pyxdf
import matplotlib.pyplot as plt
f_name = 'C:/Users/cowth/OneDrive/Dokumenter/dataset_test_psychopy/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-Default_run-001_eeg'
f_format = '.xdf'
fname = f_name + f_format
streams, header = pyxdf.load_xdf(fname)
markers0 = streams[0]["time_series"]
#markers1 = streams[1]["time_series"]
time_markers0 = streams[0]["time_stamps"]
#time_markers1 = streams[1]["time_stamps"]
#pcg_data = pcg_data/100000
print(streams[0]["info"].keys())
sfreq = float(streams[0]["info"]["nominal_srate"][0])

print(len(time_markers0), len(markers0))
print(markers0[100:150])