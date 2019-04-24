#!/usr/bin/env python
# coding: utf-8

import numpy as np

from ..source_estimate import SourceEstimate, VolSourceEstimate
from ..source_space import _ensure_src
from ..utils import check_random_state, warn, _check_option


class SourceSimulator():
    """
    Simulate Stcs
    """

    def __init__(self, tmin=None, tstep=None, subject=None, verbose=None):
        self.tmin = tmin
        self.tstep = tstep
        self.subject = subject
        self.verbose = verbose
        self.labels = []
        self.waveforms = []
        self.events = np.empty((0, 3))
        self.duration = 0
        self.slast = []

    def add_data(self, source_label, waveform, events):
        '''
        '''
        # Check for mistakes
        # Source_labels is a Labels instance
        if not isinstance(source_label, Label):
            raise ValueError('source_label must be a Label,'
                             'not %s' % type(source_label))
        # Waveform is a np.array or list of np arrays
        # If it is not a list then make it one
        if not isinstance(waveform, list) or len(waveform) == 1:
            waveform = [waveform]*len(events)
            # The length is either equal to the length of events, or 1
        if len(waveform) != len(events):
            raise ValueError('Number of waveforms and events should match'
                             'or there should be a single waveform')
        # Update the maximum duration possible based on the events
        # imax = np.argmax(events[:,2])
        # if events[imax,2]+len(waveform)
        self.labels.extend([source_label]*len(events))
        self.waveforms.extend(waveform)
        self.events = np.vstack([self.events, events])
        # First sample per waveform is the first column of events
        # Last is computed below
        self.last_sample = np.array([self.events[i, 0]+len(w)
                               for i, w in enumerate(self.waveforms)])

    def generate_stc(self, src, duration=None):
        '''
        '''
        # Duration of the simulation can be optionally provided
        # If not, the precomputed maximum last sample is used
        if duration is None:
            duration = int(np.max(self.last_sample))
        # Arbitrary chunk size, can be modified later to something else
        chunk_sample_size = min(int(1. / self.tstep), duration)
        # Loop over chunks of 1 second - or, maximum sample size.
        # Can be modified to a different value.
        for chk_start_sample in range(int(self.tmin/self.tstep), duration, chunk_sample_size):
            chk_end_sample = min(chk_start_sample+chunk_sample_size, duration)
            # Initialize the stc_data array
            stc_data_chunk = np.zeros((len(self.labels), chunk_sample_size))
            # Select only the indices that have events in the time chunk
            ind = np.nonzero(np.logical_and(self.last_sample > chk_start_sample,
                                            self.events[:, 0] < chk_end_sample))[0]
            # print(ind, ind[0])
            # Loop only over the items that are in the time chunk
            subset_waveforms = [self.waveforms[i] for i in ind]
            for i, (waveform, event) in enumerate(zip(subset_waveforms,
                                                      self.events[ind])):
                # We retrieve the first and last sample of each waveform
                # According to the corresponding event
                wf_begin = event[0]
                wf_end = self.last_sample[ind[i]]
                # Recover the indices of the event that should be in the chunk
                wf_ind = np.in1d(np.arange(wf_begin, wf_end),
                                 np.arange(chk_start_sample, chk_end_sample))
                # Recover the indices of the chunk that correspond to the overlap
                chunk_ind = np.in1d(np.arange(chk_start_sample, chk_end_sample),
                                    np.arange(wf_begin, wf_end))
                # add the resulting waveform chunk to the corresponding label
                stc_data_chunk[ind[i]][chunk_ind] += waveform[wf_ind]
            stc_chunk = simulate_stc(src, self.labels, stc_data_chunk,
                                     chk_start_sample*self.tstep,
                                     self.tstep)
            # Maybe we need to return something different for events
            yield (stc_chunk, self.events[ind])

