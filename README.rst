This package contains implementations of subspace estimation and tracking techniques, with a focus on those techniques that are compatible with missing data.

Currently implemented are:

* GROUSE
* PETRELS

Example Usage (fixed subspace)
==============================
.. code:: python

  import numpy as np

  from subtracking import StaticSubspace, Grouse, Petrels, calc_subspace_proj_error


  dg = StaticSubspace(50, 5)
  tracker = Petrels(50, 5, .98 )

  ob_count = 10000

  for i in range(ob_count):
      sampling_vec = np.ones((50,1))
      # set 4 random entries to be missing
      for i in range(4):
          idx = np.random.randint(0, high=50)
          sampling_vec[idx,0] = 0

      observation = dg.next_observation()
      tracker.consume(observation, sampling_vec)

      error = calc_subspace_proj_error(dg.U, tracker.U)

      print(error)
