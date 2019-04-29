
"""
=========================================
Simulate raw data from the sample dataset
=========================================

This example illustrates how to generate source estimates and simulate raw data
from the sample dataset using the :class:`mne.simulation.SourceSimulator' class.
Once the raw data is simulated, generated source estimates are reconstructed 
using Dynamic statistical parametric mapping (dSPM) inverse operator. 
"""

# Author: Ivana Kojcic <ivana.kojcic@gmail.com>
#         Eric Larson <larson.eric.d@gmail.com>
#         Kostiantyn Maksymenko <kostiantyn.maksymenko@gmail.com>
#         Samuel Deslauriers-Gauthier <sam.deslauriers@gmail.com>

# License: BSD (3-clause)

import os.path as op

import numpy as np

import mne
from mne.datasets import sample

print(__doc__)

# To simulate the sample dataset, information of the sample subject needs to be
# loaded. This step will download the data if it not already on your machine. 
# Subjects directory is also set so it doesn't need to be given to functions.
data_path = sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
subject = 'sample'

# First, we get an info structure from the sample subject.
evoked_fname = op.join(data_path, 'MEG', subject, 'sample_audvis-ave.fif')
info = mne.io.read_info(evoked_fname)
tstep = 1 / info['sfreq']

# To simulate sources, we also need a source space. It can be obtained from the
# forward solution of the sample subject.
meg_path = op.join(data_path, 'MEG', subject)
fwd_fname = op.join(meg_path, 'sample_audvis-meg-eeg-oct-6-fwd.fif')
fwd = mne.read_forward_solution(fwd_fname)
src = fwd['src']


################################################################################

# Load the real raw data and corresponding events.
raw = mne.io.read_raw_fif(op.join(meg_path, 'sample_audvis_raw.fif'))
raw.set_eeg_reference(projection=True).crop(0, 60)  # for speed

# Standard sample event IDs. These values will correspond to the third column
# in the events matrix.
event_id = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
            'visual/right': 4, 'smiley': 5, 'button': 32}
# Events from the experiment.
events = mne.find_events(raw)

# ##############################################################################

# In order to simulate source time courses, labels of desired active regions
# need to be specified for each of the 4 simulation conditions.
# Make a dictionary that maps conditions to activation strengths within
# aparc.a2009s labels. In the aparc.a2009s parcellation:
#   -  'G_temp_sup-G_T_transv' is the label for primary auditory area 
#   -  'S_calcarine' is the label for primary visual area
# In each of the 4 conditions, only the primary area is activated. This means
# that during the activations of auditory areas, there are no activations in
# visual areas and vice versa.
# Moreover, for each condition, contralateral region is more active (here, 2
# times more) than the ipsilateral.

activations = {
    'auditory/left':
        [('G_temp_sup-G_T_transv-lh', 100),          # label, activation (nAm)
         ('G_temp_sup-G_T_transv-rh', 200)],
    'auditory/right':
        [('G_temp_sup-G_T_transv-lh', 200),
         ('G_temp_sup-G_T_transv-rh', 100)],
    'visual/left':
        [('S_calcarine-lh', 100),
         ('S_calcarine-rh', 200)],
    'visual/right':
        [('S_calcarine-lh', 200),
         ('S_calcarine-rh', 100)],
}

annot='aparc.a2009s'

# Load the 4 necessary label names.
label_names = sorted(set(activation[0]
                         for activation_list in activations.values()
                         for activation in activation_list))
region_names = list(activations.keys())

#  Define the time course of the activity for each region to activate. We use a
#  sine wave and it will be the same for all 4 regions.
source_time_series = np.sin(np.linspace(0, 4 * np.pi, 100)) * 10e-7


# Create simulated source activity.

# Here, SourceSimulator is used, which allows to specifiy where (label), what
# (source_time_series), and when (events) event type will occur.

# We will add data for 4 areas, each of which contains 2 labels. Since add_data
# method accepts 1 label per call, it will be called 2 times per area.
# All activations will contain the same waveform, but the amplitude will be 2
# times higher in the contralateral label, as explained before.

# When the activity occurs is defined using events. In this case, they are taken
# from the original raw data. The first column is the sample of the event, the
# second is not used. The third one is the event id, which is different for each
# of the 4 areas.

region_id = 0
source_simulator = mne.simulation.SourceSimulator(src, tstep=tstep)
for i, region_name in enumerate(region_names):
    region_id = i+1
    for i in range(2):
        label_name = activations[region_name][i][0]
        label_tmp = mne.read_labels_from_annot(subject, annot,
                                               subjects_dir=subjects_dir,
                                               regexp=label_name,verbose=False)
        label_tmp = label_tmp[0]
        amplitude_tmp = activations[region_name][i][1]  
        events_tmp = events[np.where(events[:,2]== region_id)[0],:]
        source_simulator.add_data(label_tmp, amplitude_tmp*source_time_series,
                                  events_tmp)

# To obtain a SourceEstimate object, we need to use `get_stc()` method of
# SourceSimulator class.
stc_data = source_simulator.get_stc()


# Simulate raw data.

# Project the source time series to sensor space. Three types of noise will be
# added to the simulated raw data:
#     - multivariate Gaussian noise computed using a noise covariance from
#       original epoched data
#     - blink (EOG) noise
#     - ECG noise
# The source simulator can be given directly to the `simulate_raw` function.

raw_sim = mne.simulation.simulate_raw(info, source_simulator, forward=fwd,
                                      cov=None)
noise_epochs = mne.Epochs(raw, events, tmax=0)
noise_epochs.info['bads'] = []
noise_cov = mne.compute_covariance(noise_epochs)

mne.simulation.add_noise(raw_sim, cov=noise_cov, random_state=0)
mne.simulation.add_eog(raw_sim, random_state=0)
mne.simulation.add_ecg(raw_sim, random_state=0)

# Plot original and simulated raw data.
raw.plot()
raw_sim.plot(title='Simulated raw data from sample dataset')


# Reconstruct simulated source time courses using dSPM inverse operator.

# Here, source time courses for auditory and visual areas are reconstructed
# separately and their difference is shown. This was done merely for better
# visual representation of source recontruction.
# As expected, when high activations appear in primary auditory areas, primary
# visual areas will have low activations and vice versa.
method, lambda2 = 'dSPM', 1. / 9.
epochs = mne.Epochs(raw_sim, events, event_id)
inv = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, noise_cov)
stc_aud = mne.minimum_norm.apply_inverse(
    epochs['auditory/left'].average(), inv, lambda2, method)
stc_vis = mne.minimum_norm.apply_inverse(
    epochs['visual/right'].average(), inv, lambda2, method)
stc_diff = stc_aud - stc_vis

brain = stc_diff.plot(subjects_dir=subjects_dir, initial_time=0.1, hemi='split', 
                     title='Difference between reconstructed stcs of auditory'
                           ' and visual area',  views=['lat','med'])