# Copyright 2015 Gregory Hasseler
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc

import numpy as np


class Tracker(object, metaclass=abc.ABCMeta):
    """
    Abstract base class for tracker implementations.
    """

    @abc.abstractclassmethod
    def consume(self, ob_vec, sample_vec):
        """Consume an observation and corresponding sampling vector.

        Keyword arguments:
        ob_vec -- the observation vector with unobserved attributes not present (NumPy vector)
        sample_vec -- the sampling vector (NumPy vector)
        """

        raise NotImplementedError("users must define consume to use this base class")

    def _project(self, ob_vec, sample_vec):
        """Find the projection of the observation vector (ob_vec) into the subspace.
        Keyword arguments:
        ob_vec: observation vector with possibly missing data. Noise entries should be marked in sample_vec
        sample_vec: vector indicating missing data entries in ob_vec. Entries corresponding with ob_vec should be 1 where data was recorded and 0 where data was missing
        """

        # Take only the rows of U that correspond to attributes we have observed
        row_indices = np.nonzero(sample_vec)[0]
        U_samp = self.U[row_indices, :]
        ob_vec_samp = ob_vec[row_indices, :]

        # Compute the weights
        projection = np.linalg.lstsq(U_samp, ob_vec_samp)[0]

        return projection
