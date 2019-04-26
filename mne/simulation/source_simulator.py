#!/usr/bin/env python
# coding: utf-8

import numpy as np

from ..source_estimate import SourceEstimate, VolSourceEstimate
from ..source_space import _ensure_src
from ..utils import check_random_state, warn, _check_option
from mne.simulation import simulate_stc
from mne import Label


class SourceSimulator():
    """Class to generate simulated Source Estimates.
    Parameters
    ----------
    src : instance of SourceSpaces
        Source space.
    tstep : float
        Time step between successive samples in data. Default is 0.001 sec.
    duration : float | None
        Time interval during which the simulation takes place in seconds.
        Default value is computed using existing events and waveform lengths.
    Attributes
    ----------
    duration : float
        The duration of the simulation in seconds.
    """

    def __init__(self, src, tstep=1e-3, duration=None):
        self._src = src
        self._tstep = tstep
        self._labels = []
        self._waveforms = []
        self._events = np.empty((0, 3), dtype=int)
        self._duration = duration
        self._last_samples = []
        self._chk_duration = 1000

    @property
    def duration(self):
        """Duration of the simulation"""
        # If not, the precomputed maximum last sample is used
        if self._duration is None:
            return np.max(self._last_samples) * self._tstep
        return self._duration

    @property
    def nb_samples(self):
        """Number of samples in the simulation"""
        return int((self.duration / self._tstep)+1)

    def add_data(self, label, waveform, events):
        """Add data to the simulation
        Data should be added in the form of a triplet of
        Label (Where) - Waveform(s) (What) - Event(s) (When)
        Parameters
        ----------
        label : Label
            The label (as created for example by mne.read_label). If the label
            does not match any sources in the SourceEstimate, a ValueError is
            raised.
        waveform : list | array
            The waveform(s) describing the activity on the label vertices.
            If list, must have the same length as events
        events: array of int, shape (n_events, 3)
            Events associated to the waveform(s) to specify when the activity
            should occur.
        """
        if not isinstance(label, Label):
            raise ValueError('label must be a Label,'
                             'not %s' % type(label))

        # If it is not a list then make it one
        if not isinstance(waveform, list) or len(waveform) == 1:
            waveform = [waveform] * len(events)
            # The length is either equal to the length of events, or 1
        if len(waveform) != len(events):
            raise ValueError('Number of waveforms and events should match '
                             'or there should be a single waveform')
        # Update the maximum duration possible based on the events
        self._labels.extend([label] * len(events))
        self._waveforms.extend(waveform)
        self._events = np.vstack([self._events, events])
        # First sample per waveform is the first column of events
        # Last is computed below
        self._last_samples = np.array([self._events[i, 0] + len(w)
                                      for i, w in enumerate(self._waveforms)])

    def get_stim_channel(self, start_sample=0, nb_samples=None):
        """Get the stim channel from the provided data.
        Returns the stim channel data according to the simulation parameters
        which should be added through function add_data. If both start_sample
        and nb_samples are not specified, the entire duration is used.
        Parameters
        ----------
        start_sample : int
            First sample in chunk. Default is 0.
        nb_samples : int
            Number of samples in the stc data. If both start_sample and
            nb_samples are not specified, the entire duration is used.
        Returns
        -------
        stim_data : 1-d array of int
            The stimulation channel data.
        """
        if nb_samples is None:
            nb_samples = self.nb_samples - start_sample

        end_sample = start_sample + nb_samples
        # Initialize the stim data array
        stim_data = np.zeros(nb_samples, dtype=int)

        # Select only events in the time chunk
        stim_ind = np.where(np.logical_and(self._events[:, 0] >= start_sample,
                                           self._events[:, 0] < end_sample))[0]

        if len(stim_ind) > 0:
            relative_ind = self._events[stim_ind, 0].astype(int) - start_sample
            stim_data[relative_ind] = self._events[stim_ind, 2]

        return stim_data

    def get_stc(self, start_sample=0, nb_samples=None):
        """Simulate a SourceEstimate from the provided data.
        Returns a SourceEstimate object constructed according to the simulation
        parameters which should be added through function add_data. If both
        start_sample and nb_samples are not specified, the entire duration is
        used.
        Parameters
        ----------
        start_sample : int
            First sample in chunk. Default is 0.
        nb_samples : int
            Number of samples in the stc data. If the number of samples is not
            provided, it will contain the remaining samples after start_sample.
        Returns
        -------
        stc : SourceEstimate object
            The generated source time courses.
        """
        if len(self._labels) == 0:
            raise ValueError('No simulation parameters were found. Please use '
                             'function add_data to add simulation parameters.')
        if nb_samples is None:
            nb_samples = self.nb_samples - start_sample

        end_sample = start_sample + nb_samples
        # Initialize the stc_data array
        stc_data = np.zeros((len(self._labels), nb_samples))

        # Select only the indices that have events in the time chunk
        ind = np.where(np.logical_and(self._last_samples >= start_sample,
                                      self._events[:, 0] < end_sample))[0]
        # Loop only over the items that are in the time chunk
        subset_waveforms = [self._waveforms[i] for i in ind]
        for i, (waveform, event) in enumerate(zip(subset_waveforms,
                                                  self._events[ind])):
            # We retrieve the first and last sample of each waveform
            # According to the corresponding event
            wf_begin = event[0]
            wf_end = self._last_samples[ind[i]]
            # Recover the indices of the event that should be in the chunk
            waveform_ind = np.in1d(np.arange(wf_begin, wf_end),
                                   np.arange(start_sample, end_sample))
            # Recover the indices that correspond to the overlap
            stc_ind = np.in1d(np.arange(start_sample, end_sample),
                              np.arange(wf_begin, wf_end))
            # add the resulting waveform chunk to the corresponding label
            stc_data[ind[i]][stc_ind] += waveform[waveform_ind]
        stc = simulate_stc(self._src, self._labels, stc_data,
                           start_sample * self._tstep, self._tstep,
                           allow_overlap=True)

        return stc

    def __iter__(self):
        # Arbitrary chunk size, can be modified later to something else
        # Loop over chunks of 1 second - or, maximum sample size.
        # Can be modified to a different value.
        nb_samples = self.nb_samples
        for start_sample in range(0, nb_samples, self._chk_duration):
            chk_duration = min(self._chk_duration, nb_samples - start_sample)
            yield (self.get_stc(start_sample, chk_duration),
                   self.get_stim_channel(start_sample, chk_duration))