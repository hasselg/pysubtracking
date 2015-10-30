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

import numpy as np

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
