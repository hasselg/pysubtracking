# Copyright 2015 Gregory Hasseler
# 
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

import random

import numpy as np

class StaticSubspace:
    """
    Class to generate observations from a known static subspace.
    """
    
    def __init__(self, ambient_dim, width, rank):
        """
        Arguments:
        ambient_dim -- the ambient dimension of the observations to generate
        width -- the number of observations that should be present in the observation matrix
        rank -- the rank of the subspace the observations will belong to
        """
        
        self.ambient_dim = ambient_dim
        self.width = width
        self.rank = rank
        
        # Generate U and _V
        self.U = np.random.rand(ambient_dim, rank)
        self._V = np.random.rand(width, rank)
        
        # Generate the observation matrix X
        self.X = self.U @ self._V.T
        
    def next_observation(self):
        """
        Returns at random an observation from the static observation matrix
        """
        
        idx = random.randint(0, self.width - 1)
        ob = self.X[:,idx].reshape((self.ambient_dim, 1))

        return ob
        