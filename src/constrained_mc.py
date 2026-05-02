import numpy as np
import environment
import policy
import agent
import sys
import perseus_redux as perseus_redux
import random
from datetime import datetime
import pickle
from copy import copy
import os
from timeit import default_timer as timer
import utils
from scipy.stats import entropy
from scipy.special import softmax

def get_random_data(conc,tmax,pos):
    tstart=np.random.choice(conc.shape[0])
    if random.random()<0.5:
        data=conc[:,:,::-1]
        return data[(tstart+np.arange(tmax+1))%conc.shape[0],...]
    data=conc.copy()
    return data[(tstart+np.arange(tmax+1))%conc.shape[0],...]

def initialize_belief_and_source(ag,env,force_obs=None,force_source=None,min_radius=None):
    if min_radius is not None:
        x0=ag.true_pos[0]
        y0=ag.true_pos[1]
        for i in range(-min_radius,min_radius+1):
            for j in range(-min_radius,min_radius+1):
                if i**2+j**2<=min_radius**2 and i+x0<ag.belief.shape[0] and j+y0<ag.belief.shape[1] and i+x0>=0 and j+y0>=0:
                    ag.belief[x0+i,y0+j]=0
        ag.belief/=np.sum(ag.belief)
    if force_source is not None:
        xcoord,ycoord=force_source
    else:
        index=np.random.choice(env.dims[0]*env.dims[1],p=ag.belief.flatten())
        xcoord,ycoord=np.unravel_index(index,env.dims)
    env.set_pos(xcoord,ycoord)


import copy

def constrained_mc(
    ag,
    env,
    conc,
    n_trials=50,
    ag_start=np.array([122, 13]),
    tmax_pre=500,            # length of the natural rollout per trial
    tmax_loss=100,           # length of forced loss after each hit
    max_branches=None,       # cap total number of branched loss trajectories (None = no cap)
    max_hits_per_trial=None, # cap number of hit-onsets per trial (None = no cap)
    verbose=False,
    corr_aware=False,
    sim_r0=np.array([0, 13]),
    min_radius=2,
    initial_belief=None,
    require_not_found=True,  # if True, stop/ignore once agent is within min_radius (found)
):
    """
    Scheme (2): sample plume-loss onsets from *generic hit times* within natural rollouts.

    Returns:
        trajs: list of branched loss trajectories (each is list of positions)
        weights: uniform weights over branched trajectories
        log_nohit_weights: array (N_branches, tmax_loss) of log P(o=0 | a_t, s*)
        xhats, yhats, sigmaxs, sigmays: onset belief summaries (at branch start)
        t_offs: onset times (within the pre-rollout) for each branch
        trial_ids: which trial each branch came from
    """

    trajs = []
    log_nohit_weights_list = []
    xhats, yhats, sigmaxs, sigmays = [], [], [], []
    t_offs = []
    trial_ids = []

    xs = np.arange(0, env.dims[0])
    ys = np.arange(0, env.dims[1])

    n_branches = 0

    for i in range(n_trials):
        if max_branches is not None and n_branches >= max_branches:
            break

        env.reset()
        ag.reset(ag_start)
        initialize_belief_and_source(ag, env)

        data = get_random_data(conc, tmax=tmax_pre, pos=None)
        env.set_data(data, data_r0=np.array(sim_r0))

        if initial_belief is not None:
            ag.belief = initial_belief

        hits_this_trial = 0

        for tt in range(tmax_pre):
            if require_not_found:
                if (ag.true_pos[0] - env.x0) ** 2 + (ag.true_pos[1] - env.y0) ** 2 <= min_radius ** 2:
                    # stop the pre-rollout once found; onsets after this are not meaningful
                    break

            # Sample observation using *generative* likelihood w.r.t. true source
            p_hit = env.get_likelihood(ag.true_pos[0] - env.x0, ag.true_pos[1] - env.y0, 1)
            o = 1 if (np.random.random() < p_hit) else 0

            # Advance one step, forcing the sampled observation into the agent
            ag.stepInTime(force_obs=o, corr_aware=corr_aware)
            env.stepInTime()

            # If a hit occurred, create a branch starting from the post-hit state
            if o == 1:
                hits_this_trial += 1

                # (Optional) ignore onsets that are already essentially "found"
                if require_not_found:
                    if (ag.true_pos[0] - env.x0) ** 2 + (ag.true_pos[1] - env.y0) ** 2 <= min_radius ** 2:
                        # you can either break or continue; breaking matches "found ends episode"
                        break

                # Snapshot onset summaries (post-hit, pre-loss)
                xhat = np.sum((xs[:, None] - ag.true_pos[0]) * ag.belief)
                yhat = np.sum((ys[None, :] - ag.true_pos[1]) * ag.belief)
                sigx = np.sqrt(np.sum((xs[:, None] - ag.true_pos[0] - xhat) ** 2 * ag.belief))
                sigy = np.sqrt(np.sum((ys[None, :] - ag.true_pos[1] - yhat) ** 2 * ag.belief))

                # Deepcopy agent + env for the loss branch
                ag_b = copy.deepcopy(ag)
                env_b = copy.deepcopy(env)

                # Run forced-loss branch: force o=0 for tmax_loss steps
                traj = [ag_b.true_pos.copy()]
                logw = np.zeros((tmax_loss,), dtype=float)

                for t in range(tmax_loss):
                    logw[t] = np.log(
                        env_b.get_likelihood(ag_b.true_pos[0] - env_b.x0, ag_b.true_pos[1] - env_b.y0, 0) + 1e-12
                    )
                    ag_b.stepInTime(force_obs=0, corr_aware=corr_aware)
                    env_b.stepInTime()
                    traj.append(ag_b.true_pos.copy())

                trajs.append(traj)
                log_nohit_weights_list.append(logw)
                xhats.append(xhat); yhats.append(yhat); sigmaxs.append(sigx); sigmays.append(sigy)
                t_offs.append(tt)          # onset time within pre-rollout (the step when the hit occurred)
                trial_ids.append(i)

                n_branches += 1
                if max_branches is not None and n_branches >= max_branches:
                    break

                if max_hits_per_trial is not None and hits_this_trial >= max_hits_per_trial:
                    break

        # end pre-rollout loop
    # end trial loop

    N = len(trajs)
    if N == 0:
        # keep return types consistent
        return [], np.array([]), np.zeros((0, tmax_loss)), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    weights = np.ones((N,), dtype=float) / N
    log_nohit_weights = np.vstack(log_nohit_weights_list)
    xhats = np.array(xhats); yhats = np.array(yhats); sigmaxs = np.array(sigmaxs); sigmays = np.array(sigmays)
    t_offs = np.array(t_offs); trial_ids = np.array(trial_ids)

    return trajs, weights, log_nohit_weights, xhats, yhats, sigmaxs, sigmays, t_offs, trial_ids

def _constrained_mc(ag,env,conc,n_trials=100,ag_start=np.array([122,13]),tmax=500,verbose=False,corr_aware=False,sim_r0=np.array([0,13]),min_radius=2,initial_belief=None):
    trajs=[]
    weights=np.ones((n_trials,))
    log_nohit_weights=np.zeros((n_trials,tmax))
    xhats=np.zeros((n_trials,))
    yhats=np.zeros((n_trials,))
    sigmaxs=np.zeros((n_trials,))
    sigmays=np.zeros((n_trials,))
    xs=np.arange(0,env.dims[0])
    ys=np.arange(0,env.dims[1])
    for i in range(n_trials):
        env.reset()
        ag.reset(ag_start)
        initialize_belief_and_source(ag,env)
        #print('source at',[env.x0,env.y0])
        data=get_random_data(conc,tmax=500,pos=None)
        env.set_data(data,data_r0=np.array(sim_r0))
        if initial_belief is not None:
            ag.belief=initial_belief
        found=False
        tt=0
        while True:
            p = env.get_likelihood(ag.true_pos[0]-env.x0,ag.true_pos[1]-env.y0,1)
            if np.random.random() < p :
                break
         #   print(ag.true_pos)
            ag.stepInTime(make_obs=True,corr_aware=corr_aware)
            env.stepInTime()
            if (ag.true_pos[0]-env.x0)**2+(ag.true_pos[1]-env.y0)**2<=min_radius**2:
                found=True
                break
            tt+=1
            if tt==500:
                break
        if found or tt==500:
            #print('source found')
            weights[i]=0
            trajs.append([])
        else:
            #print('forcing no hit')
            weights[i]=1
            traj=[ag.true_pos.copy()]
            xhats[i]=np.sum((xs[:,None]-ag.true_pos[0])*ag.belief)
            yhats[i]=np.sum((ys[None,:]-ag.true_pos[1])*ag.belief)
            sigmaxs[i]=np.sqrt(np.sum((xs[:,None]-ag.true_pos[0]-xhats[i])**2*ag.belief))
            sigmays[i]=np.sqrt(np.sum((ys[None,:]-ag.true_pos[1]-yhats[i])**2*ag.belief))
            for t in range(tmax):
                if t==0:
                    ag.stepInTime(force_obs=1,corr_aware=corr_aware)
                else:
                    #print(ag.true_pos)
                    log_nohit_weights[i,t]=np.log(env.get_likelihood(ag.true_pos[0]-env.x0,ag.true_pos[1]-env.y0,0)+1e-12)
                    ag.stepInTime(force_obs=0,corr_aware=corr_aware)
                traj.append(ag.true_pos.copy())
            trajs.append(traj)
    weights/=np.sum(weights)

    return trajs,weights,log_nohit_weights,xhats,yhats,sigmaxs,sigmays

isotropic = False
if 'ISOTROPIC' in os.environ:
    isotropic=bool(int(os.environ.get('ISOTROPIC')))

ignore_errors=False
if os.getenv('IGNORE_ERRORS') is not None:
    ignore_errors=bool(int(os.environ.get('IGNORE_ERRORS')))
new_sai=None
if os.getenv('NEW_SAI') is not None:
    new_sai=bool(int(os.environ.get('NEW_SAI')))
sai_param=1/2
if os.getenv('SAI_PARAM') is not None:
    sai_param=float(os.environ.get('SAI_PARAM'))


force_no_corr=False
if os.getenv('FORCE_NO_CORR') is not None:
    force_no_corr=bool(int(os.environ.get('FORCE_NO_CORR')))

min_radius=0
if os.getenv('MIN_RADIUS') is not None:
    min_radius=int(os.environ.get('MIN_RADIUS'))

policy_file=None
policy_dir=None
#filename and directoy of the alpha-vector policy, if used
if os.getenv('POLICY_FILE') is not None:
    policy_file=os.environ.get('POLICY_FILE')
    policy_dir=os.environ.get('POLICY_DIR')

#filename and directory to dump the search trial data
data_file=os.environ.get('DATA_FILE')
data_dir=os.environ.get('DATA_DIR')

#filename and directory of concentration data, if used
conc_file=None
conc_dir=None
if os.getenv('CONC_FILE') is not None:
    conc_file=os.environ.get('CONC_FILE')
    conc_dir=os.environ.get('CONC_DIR')

threshold = None
if os.getenv('THRESHOLD') is not None:
    threshold=float(os.environ.get('THRESHOLD'))

#position of source in the data
source_x0=int(os.environ.get('SOURCE_X0'))
source_y0=int(os.environ.get('SOURCE_Y0'))

#agent start position
ag_start_x=int(os.environ.get('AG_START_X'))
ag_start_y=int(os.environ.get('AG_START_Y'))

#dimensions of the gridworld
shape_x=int(os.environ.get('SHAPE_X'))
shape_y=int(os.environ.get('SHAPE_Y'))

dummy=False
if 'DUMMY' in os.environ:
    dummy=bool(int(os.environ.get('DUMMY')))

#experimental, not typically used
min_ell=0
if os.getenv('MIN_ELL') is not None:
    min_ell=float(os.environ.get('MIN_ELL'))

obs_per_action=1
if os.getenv('OBS_PER_ACTION') is not None:
    obs_per_action=float(os.environ.get('OBS_PER_ACTION'))

#how many snapshots to increment between agent timesteps
tstep=1
if os.getenv('TSTEP') is not None:
    tstep=float(os.environ.get('TSTEP'))

env_params={
  "data":None,
  "threshold":threshold,
  "dummy":dummy,
  "tstep":tstep
}


#the policy to test. recognizes 'sarsop', 'sai','infotaxis','trivial'
policy_name=os.environ.get('POLICY_NAME')

#set to 0 if a dummy environment without correlations, or if one-step conditional likelihoods are not available or used
corr_env=True
if 'CORR_ENV' in os.environ:
    corr_env=bool(int(os.environ.get('CORR_ENV')))

#set to 1 if you want a correlation-aware policy
corr_pol = False
if 'CORR_POL' in os.environ:
    corr_pol=bool(int(os.environ.get('CORR_POL')))

ag_start=np.array([ag_start_x,ag_start_y])

env=environment.OlfactorySearch2D((shape_x,shape_y),corr=corr_env,min_ell=min_ell,**env_params)
env2=environment.OlfactorySearch2D((shape_x,shape_y),corr=corr_env,min_ell=min_ell,threshold=7e-6)
ag=agent.CorrAgent(env,ag_start,obs_per_action=obs_per_action)
ag2=agent.CorrAgent(env2,ag_start)

with open(conc_dir+'/'+conc_file,'rb') as f:
    conc=pickle.load(f)


ells=utils.get_likelihood_from_conc(conc,threshold=7e-6,isotropic=isotropic,tstep=tstep)
l0=[ells['p_right_blank'],ells['p_left_blank'],ells['p_up_blank'],ells['p_down_blank']]
l1=[ells['p_right_whiff'],ells['p_left_whiff'],ells['p_up_whiff'],ells['p_down_whiff']]
l_un=ells['p_unconditional']

env2.set_likelihood(l_un,l0,l1,sim_r0=[source_x0,source_y0])

ag2.updateBelief(1,None)
initial_belief=(ag2.belief>0).astype(float)
initial_belief/=np.sum(initial_belief)

#compute likelihood from concentration if necessary
if 'LIKELIHOOD_FILE' not in os.environ:
    conc_dir=str(os.environ.get('CONC_DIR'))
    conc_file=str(os.environ.get('CONC_FILE'))
    with open(conc_dir+'/'+conc_file,'rb') as f:
        conc_list=pickle.load(f)
    ells=utils.get_likelihood_from_conc(conc_list,threshold=threshold,tstep=tstep,isotropic=isotropic)

#location of the likelihood files. expected format is a pickled dictionary with keys (at a minimum) 'p_unconditional' and 
#(if corr_env is True) 'p_x_y',where x is right,left,up,or down (meaning upwind, downwind, and the two crosswind directionss respectively)
# and y is blank or whiff, indicating the previous observation on which to condition.
# if corr_env is False, the code also accepts a pickled numpy array with the unconditional likelihood
else:
    likelihood_dir=os.environ.get('LIKELIHOOD_DIR')
    likelihood_file=os.environ.get('LIKELIHOOD_FILE')

    with open(likelihood_dir+'/'+likelihood_file,'rb') as f:
        ells=pickle.load(f)

if corr_env:
    if force_no_corr:
        l0=[ells['p_unconditional'] for i in range(4)]
        l1=[ells['p_unconditional'] for i in range(4)]
    else:
        l0=[ells['p_right_blank'],ells['p_left_blank'],ells['p_up_blank'],ells['p_down_blank']]
        l1=[ells['p_right_whiff'],ells['p_left_whiff'],ells['p_up_whiff'],ells['p_down_whiff']]
    l_un=ells['p_unconditional']
    l_static_0=None
    l_static_1=None
    if 'p_static_blank' in ells:
        l_static_0=ells['p_static_blank']
        l_static_1=ells['p_static_whiff']
    env.set_likelihood(l_un,l0,l1,sim_r0=[source_x0,source_y0],l_static_0=l_static_0,l_static_1=l_static_1)
else:
    if isinstance(ells,dict):
        ells=ells['p_unconditional']
    l_un=ells
    env.set_likelihood(l_un,sim_r0=[source_x0,source_y0])
print(policy_name)
if policy_file is not None:
    print(policy_file)

#experimental, not typically used
exponents=None
if "DELTA_EXPONENT" in os.environ:
    if bool(int(os.environ.get('DELTA_EXPONENT'))):
        exponents=[1-np.sum((l1[i]-l0[i])*ells['p_unconditional'])/np.sum(ells['p_unconditional']) for i in range(4)]
        for e in exponents:
            assert not np.isnan(e)
    else:
        exponents=None
if "FLOAT_EXPONENT" in os.environ:
    e=float(os.environ.get('FLOAT_EXPONENT'))
    exponents=[e,e,e,e]
print('using Bayesian learning exponents',exponents)

if policy_name=='sarsop' and not corr_pol:
    with open(policy_dir+'/'+policy_file,'rb') as f:
        alphas=pickle.load(f)
    vf0=perseus_redux.ValueFunction(env)
    vf0.load_alphas(alphas)
    pol=policy.OptimalPolicy(vf0,ag,parallel=True)
elif policy_name=='sarsop' and corr_pol:
    with open(policy_dir+'/'+policy_file,'rb') as f:
        alphas=pickle.load(f)
    vf0=perseus_redux.ValueFunction(env)
    vf1=perseus_redux.ValueFunction(env)
    vf0.load_alphas(alphas['alphas_0'])
    vf1.load_alphas(alphas['alphas_1'])
    pol=policy.OptimalPolicyWithCorr(vf0,vf1,ag,parallel=True)
elif policy_name=='sai':
    pol=policy.SpaceAwareInfotaxis(ag,with_corr=corr_pol,new=new_sai,alpha=sai_param,verbose=False)
elif policy_name=='infotaxis':
    pol=policy.InfotacticPolicy(ag,with_corr=corr_pol,verbose=False,exponents=exponents)
elif policy_name=='trivial':
    pol=policy.TrivialPolicy()

else:
    raise RuntimeError('name not recognized')


#the number of trials to perform
ag.set_policy(pol)


conc_dir=str(os.environ.get('CONC_DIR'))
conc_file=str(os.environ.get('CONC_FILE'))
with open(conc_dir+'/'+conc_file,'rb') as f:
    conc_list=pickle.load(f)

env2=environment.OlfactorySearch2D((shape_x,shape_y),dummy=False,threshold=7e-6,corr=True)
env2.set_likelihood(l_un,l0,l1,sim_r0=np.array([source_x0,source_y0]))
ag2=agent.CorrAgent(env2,np.array([ag_start_x,ag_start_y]))

ag2.updateBelief(1,None)
locs=[]
for i in range(-2,3):
    for j in range(-2,3):
        if i**2+j**2<=2**2:
            locs.append(ag2.true_pos+np.array([i,j]))
ag2.zero_out_locs(locs)

initial_belief=ag2.belief.copy()
initial_belief=(initial_belief>0).astype(float)
initial_belief/=np.sum(initial_belief)

#t1=int(os.environ.get('T1'))

n_trials=int(os.environ.get('N_TRIALS'))
horizon=int(os.environ.get('HORIZON'))
gamma=float(os.environ.get('GAMMA'))

force_source_x=int(os.environ.get('FORCE_SOURCE_X'))
force_source_y=int(os.environ.get('FORCE_SOURCE_Y'))


trajs, weights, log_nohit_weights, xhats, yhats, sigmaxs, sigmays, t_offs, trial_ids = constrained_mc(ag,env,conc=conc_list[0],n_trials=n_trials,initial_belief=initial_belief,tmax_loss=horizon,tmax_pre=100)

data={'trajs':trajs,'weights':weights,'log_nohit_weights':log_nohit_weights,'x_hat':xhats,'y_hat':yhats,'sigma_x':sigmaxs,'sigma_y':sigmays}

with open(data_dir+'/'+data_file,'wb') as f:
    pickle.dump(data,f)


    
