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

import numpy as np
import scipy.linalg as splinalg


class StaticSubspace:
    """
    Class to generate observations from a known static subspace.
    """

    def __init__(self, ambient_dim, rank):
        """
        Arguments:
        ambient_dim -- the ambient dimension of the observations to generate
        rank -- the rank of the subspace the observations will belong to
        """

        self.ambient_dim = ambient_dim
        self.rank = rank

        # Generate U
        self.U = np.random.rand(ambient_dim, rank)

    def next_observation(self):
        """
        Returns at random an observation from the static observation matrix
        """

        random_projection = np.random.rand(self.rank, 1)
        ob = self.U @ random_projection

        return ob


class RotatingSubspace:
    """
    Class to generate observations from a rotating subspace.
    """

    def __init__(self, ambient_dim, rank, delta=1E-5):
        """
        Arguments:
        ambient_dim -- the ambient dimension of the observations to generate
        rank -- the rank of the subspace the observations will belong to
        delta --
        """

        self.ambient_dim = ambient_dim
        self.rank = rank
        self.delta = delta
        self.ob_count = 0

        # Generate initial U as a normal random matrix
        self._U_0 = splinalg.orth(np.random.randn(ambient_dim, rank))

        # Generate skew symmetric rotation matrix
        B = np.zeros((self.ambient_dim, self.ambient_dim))
        idcs = np.tril_indices(self.ambient_dim, k=-1)
        r_count = len(idcs[0])
        rands = np.random.randn(r_count)
        B[idcs] = rands
        self._B = B + (-1 * B.T)

    def next_observation(self):
        """
        Rotates the internal subspace and returns a new random observation from it.
        """

        self.ob_count += 1

        # Rotate the subspace
        self.U = splinalg.expm(self.ob_count * self.delta * self._B) @ self._U_0

        # Generate a random observation
        random_projection = np.random.rand(self.rank, 1)
        ob = self.U @ random_projection

        return ob
