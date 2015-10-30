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

def calc_subspace_proj_error(U, U_hat):
    """Calculate the normalized projection error between two orthogonal subspaces.
    Keyword arguments:
    U: ground truth subspace
    U_hat: estimated subspace
    """
    I = np.identity(U.shape[0])
    top = np.linalg.norm(I - U_hat @ U_hat.T, ord="fro")
    bottom = np.linalg.norm(U, ord="fro")
    
    error = float(top) / float(bottom)
    
    return error

def calc_matrix_error(X, X_hat):
    """Calculate the normalized error between two matrices.
    Keyword arguments:
    X: ground truth matrix
    X_hat: estimated matrix
    """
    top = np.linalg.norm(X - X_hat, ord="fro")
    bottom = np.linalg.norm(X, ord="fro")
    
    error = float(top) / float(bottom)
    
    return error

def calc_observation_error(x, x_hat):
    """Calculate the normalized error between two vectors.
    Keyword arguments:
    x: ground truth observation
    x_hat: estimated observation
    """
    top = np.linalg.norm(x - x_hat, ord="fro")
    bottom = np.linalg.norm(x, ord="fro")
    
    error = float(top) / float(bottom)
    
    return error
 
