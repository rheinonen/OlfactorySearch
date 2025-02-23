# OlfactorySearch (v0.1)
Tools for simulating the olfactory search POMDP. This is an evolution of the earlier project PerseusPOMDP. The emphasis here is on searching in concentration data taken from a DNS, but it is also possible to search in a stochastic environment with encounters artificially drawn from a likelihood. This code was used in the papers Heinonen et al. (arXiv: 2409.11343) and Heinonen et al. (arXiv: TBD). A number of changes were made before publishing this code in order to improve its accessibility, but these changes have NOT yet been fully tested.

## The environment
The olfactory search POMDP is implemented as a class in the file environment.py. This class is called `OlfactorySearch2D`. As the name suggests, this class is intended for use in a 2D environment (for our purposes, this is typically understood as a 2D slice of a larger 3D world). When instantiating, the important parameters are as follows:
- `dims`: a tuple of ints, which defines the gridworld where the search will take place. The first dimension should be along the direction of the mean wind, if present. This parameter is the only one required.
- `dummy`: bool. If True, the environment is stochastic and will draw observations from the likelihood. If False, observations will be drawn from concentration data.
- `corr`: bool. Should be set to true if you want to store information about conditional observation likelihoods, that is conditioned on the observation made at the previous timestep and the action taken. This is important if the agent is aware of 1-step Markovian correlations, or if `dummy=True` and you want to draw detections from a stochastic environment correlated at the 1-step Markov level.
- `threshold`: float. Must be set if `dummy=False.` An encounter with the odor is made if the local concentration instantaneously is equal to or exceeds this number.
Eventually, you will need to call `set_likelihood()` in order to define the likelihood of an encounter (as a 2D numpy array). If `dummy=False`, you will also need to call `set_data()` to choose the concentration data in which to search.
- `tstep`: float. This sets the agent's timestep, i.e.\ how many snapshots to skip when the agent makes an observation and takes an action. If not an integer, the concentration data will be linearly interpolated (this may be a bad approximation, especially if the wind speed is strong).

## The agent
The Bayesian searcher is implemented as the class `CorrAgent` in the file agent.py. It makes observations, updates its belief on the source location, and moves one grid space according to a policy. At a minimum, you need to set the environment and the position of the agent in the gridworld. Eventually, it will also need a policy. Optionally, you can set a parameter `belief_env` which is useful if its model is misspecified, i.e.\ the agent thinks it's in a different `OlfactorySearch2D` environment than it actually is. Also, you can set `obs_per_action` to be either an integer or a float equal to 1 over an integer, which will allow the agent to make multiple observations before taking an action, or vice versa (the default is to make one observation and take one action every timestep). 

## The policy
A policy is just a map that takes a belief and outputs an action, implemented as the class function getAction(). A number of policies are implemented in policy.py, including infotaxis (Vergassola, Villermaux, and Shraiman 2007), space-aware infotaxis (Loisy and Eloy 2022), and quasi-optimal policies represented as a set of alpha-vectors. The quasi-optimal policy must be precomputed by some method, such as the SARSOP algorithm (Kurniawati, Hsu, and Lee 2008), available [here](https://github.com/AdaCompNUS/sarsop).

## Performing search trials
A script main.py has been provided to conduct search trials. This script assumes a number of environmental variables have been set. At a minimum:
- `DATA_DIR`: where to put the search trial data.
- `DATA_FILE`: what to call the search trial data file. It will saved as a pickled dictionary.
- `TMAX`: the maximum search time. Searching beyond this time will be considered a failed search.
- `SOURCE_X0`: the x-position (i.e., along the wind) of the source in the likelihood array.
- `SOURCE_Y0` the y-position (i.e., transverse to the wind) of the source in the likelihood array.
- `SHAPE_X`: the first dimension of the gridworld.
- `SHAPE_Y`: the second dimension of the gridworld,
- `AG_START_X`: where the agent is to start its search (x coordinate). By default, the agent will start with an encounter, and the source will be randomly drawn from the prior that is accordingly induced.
- `AG_START_Y`: where the agent is to start is search (y coordinate).
- `POLICY_NAME`: the policy to test. currently recognizes 'sarsop', 'sai', 'infotaxis', 'trivial'
- `N_TRIALS`: integer. The number of search trials to perform
Also frequently set:
- `CONC_DIR`: where the concentration data is.
- `CONC_FILE`: the concentration data file.
- `LIKELIHOOD_FILE`: the encounter likelihood. This will be computed from the concentration data if not set.
- `LIKELIHOOD_DIR`: where the likelihood is located.
- `DUMMY`: 0 or 1. whether or not to use a stochastic environment (as opposed to concentration data). Defaults to 0.
- `THRESHOLD`: float. The concentration threshold. Obligatory if using concentration data.
- `POLICY_FILE`: name of the policy file, if using a quasi-optimal policy.
- `POLICY_DIR`: location of the policy file.
- `CORR_POL`: 0 or 1. whether or not the agent is aware of 1-step Markov correlations. Defaults to 0.
- `CORR_ENV`: 0 or 1. whether or not to use 1-step conditional likelihoods. must be 1 if CORR_POL is 1 or if DUMMY=1 and you want a correlated (at the 1-step level) environment. Defaults to 1.
- `SAVE_DISPLACEMENTS`: 0 or 1. turn on if you want to store the trajectories of the agent (as in our paper on optimal trajectories). Defaults to 0.
- `TSTEP`: float. The timestep for the concentration data. Defaults to 1.
- `OBS_PER_ACTION`: int or float. the number observations per action, as described in the agent section.

## Other tools
We also provide a script make_pomdp_file.py, which creates the POMDP according to a specified likelihood file or concentration data file, in Tony Cassandra's .pomdp format. Additionally, there is convert_policy.py, which converts a policy computed using SARSOP to a format which is understood by the code. 

## Sample data
Concentration data used in the trajectories paper can be found on the [Smart-TURB project website](https://smart-turb.roma2.infn.it/). Along with SARSOP and this code, this data should suffice to reproduce all results found therein. Raw Lagrangian data will be provided in the future.


