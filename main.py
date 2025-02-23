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

'''
This script tests a policy. It supports many features, including dummy environments, correlation-aware agents, and multiple detections per action (and vice-versa).
This was written for HPC resources and expects a number of environmental variables to be set.
'''

def get_random_data_from_file(file_list,tmax=2500):
    file_choice=random.choice(file_list)
    with open(file_choice,'rb') as f:
        data=pickle.load(f)
    conc=data['concentration']
    tstart=np.random.choice(conc.shape[0]-tmax+1)
    return conc[tstart:tstart+tmax,...]

def get_random_data(conc_list,good_starts,tmax=2500,isotropic=False):
    indices=random.choice(good_starts)
    
    data=conc_list[indices[0]]
    if isotropic:
        conc=data
        x=indices[2]
        y=indices[3]
        if random.random()<0.5:
            conc=np.transpose(conc,axes=(0,2,1))
            tmp=x
            x=y
            y=tmp
        if random.random()<0.5:
            conc=conc[:,::-1,:]
            x=conc.shape[1]-x-1
        if random.random()<0.5:
            conc=conc[:,:,::-1]
            y=conc.shape[2]-y-1
        tstart=indices[1]
        return conc[tstart:tstart+int(tmax*tstep)+1,...],x,y,indices[0],tstart
    if random.random()<0.5:
        conc=data[:,:,::-1]
        tstart=indices[1]
        return conc[tstart:tstart+int(tmax*tstep)+1,...],indices[2],conc.shape[2]-indices[3]-1,indices[0],tstart
    conc=data
    tstart=indices[1]
    return conc[tstart:tstart+int(tmax*tstep)+1,...],indices[2],indices[3],indices[0],tstart

def get_data_from_file(file_list,good_starts,tmax=2500):
    indices=random.choice(good_starts)
    with open(file_list[indices[0]],'rb') as f:
        data=pickle.load(f)
    if random.random()<0.5:
        conc=data['concentration'][:,:,::-1]
        tstart=indices[1]
        return conc[tstart:tstart+tmax+1,...],indices[2],conc.shape[2]-indices[3]-1,indices[0],tstart
    conc=data['concentration']
    tstart=indices[1]
    return conc[tstart:tstart+tmax+1,...],indices[2],indices[3],indices[0],tstart
    

def conditional_mean(times,tmax):
    filtered=[x for x in times if x<tmax]
    mean=np.mean(filtered)
    err=np.std(filtered)/np.sqrt(len(filtered))
    failure=1-len(filtered)/len(times)
    return mean,err,failure

def initialize_belief_and_source(ag,env,force_obs=None,force_source=None,min_radius=None):
    if force_obs is not None:
        obs=force_obs
    else:
        obs=utils.initial_hit_aurore(env)    
    ag.updateBelief(obs,None)
    ag.last_obs=1
    if min_radius is not None:
        x0=ag.true_pos[0]
        y0=ag.true_pos[1]
        for i in range(-min_radius,min_radius+1):
            for j in range(-min_radius,min_radius+1):
                if i**2+j**2<min_radius**2 and i+x0<ag.belief.shape[0] and j+y0<ag.belief.shape[1] and i+x0>=0 and j+y0>=0:
                    ag.belief[x0+i,y0+j]=0
        ag.belief/=np.sum(ag.belief)
    if force_source is not None:
        xcoord,ycoord=force_source
    else:
        index=np.random.choice(np.arange(env.dims[0]*env.dims[1],dtype='int'),p=ag.belief.flatten())
        xcoord,ycoord=np.unravel_index(index,env.dims)
    env.set_pos(xcoord,ycoord)


def policy_trials(ag,env,n_trials,conc_list=None,good_starts=None,thompson=False,tmax=1000,ag_start=[5,16],errors=False,vf=None,error_frac=0.3333,bad_traj=False,force_obs=None,min_radius=None,verbose=False,corr_aware=False,sim_r0=[93,16],isotropic=False,save_beliefs=False,ignore_errors=True,deltas=False,save_displacements=False,force_source=None,exponents=None):
    '''
    the main function called in this script. it needs an agent, an environment and a number of search trials to perform.
    the important parameters are listed below.
    - conc_list and good_starts are respectively a list of concentration data arrays to choose from and a list of indices (conc,t,x,y) which describe potential starting positions
    in the data. typically, these are where there is a hit.
    - tmax is the max search time allowed
    - ag_start is the position in the gridworld where the agent starts. the source will be decided according to an initial detection or forced.
    - force_obs is usually taken to be 1. it forces the agent to make the observation at time zero
    - corr_aware is True if the agent is aware of 1-step correlations, otherwise False
    - sim_r0 is the location of the source in the concentration data
    - isotropic is True in the absence of mean wind, otherwise False
    - force_source places the source at a fixed location, used to perform trials always from the same starting position relative to the source
    
    a number of objects are returned, by far the most important of which are times and starts, which are the arrival times and the respective source positions for each trial
    '''
    times=[]
    starts=[]
    hits=[]
    all_entropies=[]
    actions=[]
    beliefs=[]
    deltas=[]
    displacement_x=[]
    displacement_y=[]
    pos=[]
    all_hit_times=[]
    if bad_traj:
        bad_trajs=[]
    else:
        bad_trajs=None
    if errors:
        bellman=[]
    else:
        bellman=None
    for j in range(n_trials):
        if j%10==0:
            print("starting trial",j+1)
        hit_times=[]
        if env.dummy:
            if force_source is not None:
                src=force_source
            else:
                src=None    
        else:
            data,x0,y0,filename,tstart=get_random_data(conc_list,good_starts,tmax,isotropic=isotropic)
            if verbose:
                print('hit position in simulation:',x0,y0)
            env.set_data(data,data_r0=np.array(sim_r0))
            src=ag_start+np.array([sim_r0[0]-x0,sim_r0[1]-y0])
             
        env.reset()
        ag.reset(ag_start)
        if thompson:
            ag.policy.reset()
        initialize_belief_and_source(ag,env,force_obs=force_obs,force_source=src,min_radius=min_radius)
        starts.append([env.x0,env.y0])
        if verbose:
            print('source at',env.x0,env.y0)
        traj=[ag.true_pos]
        dx=[0]
        dy=[0]
        b0=ag.belief
        entropies=[entropy(b0.flatten())]
        if bad_traj or save_beliefs:
            bs=[b0]
        #print('agent at',ag.true_pos)
        for k in range(tmax):
            if errors:
                b,v=ag.stepInTime(values=True,corr_aware=corr_aware)
            else:
                if ignore_errors:
                    try: 
                        prev_pos=ag.true_pos  
                        b=ag.stepInTime(make_obs=k!=0,corr_aware=corr_aware)
                        if save_displacements:
                            dx.append(ag.true_pos[0]-ag_start[0])
                            dy.append(ag.true_pos[1]-ag_start[1])               
                        if save_deltas:
                            delta=env.get_likelihood(prev_pos[0]-env.x0,prev_pos[1]-env.y0,1,obs_state=1,action=ag.last_action)-env.get_likelihood(prev_pos[0]-env.x0,prev_pos[1]-env.y0,1,obs_state=0,action=ag.last_action)
                            deltas.append(delta)
                    except:
                        print('error in step in time, most likely zero belief. setting to tmax')
                        print('agent stuck count was',ag.stuck_count)
                        print('time was',k)
                        print('did agent interact with boundary?',ag.boundary)
                        k=tmax-1
                        break
                else:
                    b=ag.stepInTime(make_obs=k!=0,corr_aware=corr_aware,exponents=exponents)
            if bad_traj or save_beliefs:
                bs.append(ag.belief)
            if errors and random.random()<error_frac:
                bellman.append(vf.bellmanErrorNew(b,value=v))
            entropies.append(entropy(ag.belief.flatten()))
            env.stepInTime()
            actions.append(ag.last_action)
            pos.append(ag.true_pos-np.array([env.x0,env.y0])+np.array(sim_r0))
            traj.append(ag.true_pos)
            if verbose:
                print('agent at',ag.true_pos)
            if ag.last_obs and verbose:
                print('hit')
            if ag.last_obs:
                hit_times.append(k)
            if ag.true_pos[0]==env.x0 and ag.true_pos[1]==env.y0:
                hits.append(ag.nhits-1)
                break
            #if ag.stuck_count>8:
            #    k=tmax-1
            #    break
        displacement_x.append(dx)
        displacement_y.append(dy)
        times.append(k+1)
        all_entropies.append(entropies)
        all_hit_times.append(hit_times)
        if save_beliefs:
            beliefs.append(bs)
        if k==tmax-1 and bad_traj:
            bad_trajs.append([j,hit_times,traj,filename,tstart])
    return times,starts,b0,all_hit_times,bellman,bad_trajs,all_entropies,actions,beliefs,deltas,displacement_x,displacement_y,pos

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

tmax=int(os.environ.get('TMAX'))

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
ag=agent.CorrAgent(env,ag_start,obs_per_action=obs_per_action)


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
n_trials=int(os.environ.get('N_TRIALS'))
ag.set_policy(pol)
thompson=False

save_beliefs=False
if "SAVE_BELIEFS" in os.environ:
    save_beliefs=bool(int(os.environ.get('SAVE_BELIEFS')))

#set these if you want the agent to always start in the same relative position wrt the source
force_source=None
if "FORCE_SOURCE_X" in os.environ:
    force_source_x=int(os.environ.get('FORCE_SOURCE_X'))
    force_source_y=int(os.environ.get('FORCE_SOURCE_Y'))
    force_source=np.array([force_source_x,force_source_y])

print(data_file)

if dummy:
    conc_list=None
    hit_starts=None
else:
    if force_source is not None:
        hit_starts=[]
        conc_dir=str(os.environ.get('CONC_DIR'))
        conc_file=str(os.environ.get('CONC_FILE'))
        with open(conc_dir+'/'+conc_file,'rb') as f:
            conc_list=pickle.load(f)
        ag_pos=ag_start-force_source+np.array([source_x0,source_y0])
        for i,conc in enumerate(conc_list):
            indices=np.argwhere(conc[:int(conc.shape[0]-tmax*tstep),ag_pos[0],ag_pos[1]]>=threshold)
            prepended=[[i]+list(index)+list(ag_pos.astype('int')) for index in indices]
            hit_starts=hit_starts+prepended
    else:
        hit_starts=[]
        conc_dir=str(os.environ.get('CONC_DIR'))
        conc_file=str(os.environ.get('CONC_FILE'))
        with open(conc_dir+'/'+conc_file,'rb') as f:
            conc_list=pickle.load(f)
        for i,conc in enumerate(conc_list):
            indices=np.argwhere(conc[:int(conc.shape[0]-tmax*tstep),...]>=threshold)
            prepended=[[i]+list(index) for index in indices if not ((index[1]==source_x0 and index[2]==source_y0) or (index[1]-source_x0)**2+(index[2]-source_y0)**2<min_radius**2)]
            hit_starts=hit_starts+prepended
        print('there are',len(hit_starts),'acceptable starting positions in this dataset')  

if "SAVE_DELTAS" in os.environ:
    save_deltas=bool(int(os.environ.get('SAVE_DELTAS')))
else:
    save_deltas=False

#set to 1 if you want to save the Delta x and Delta y of the agent during its search
if "SAVE_DISPLACEMENTS" in os.environ:
    save_displacements=bool(int(os.environ.get('SAVE_DISPLACEMENTS')))
else:
    save_displacements=False

# for debugging can be useful to set to 1
verbose=False
if "VERBOSE" in os.environ:
    verbose=bool(int(os.environ.get('VERBOSE')))
times,starts,b0,hits,tmp,bad_traj,entropies,actions,beliefs,deltas,displacement_x,displacement_y,pos=policy_trials(ag,env,n_trials,conc_list,hit_starts,tmax=tmax,bad_traj=False,thompson=thompson,force_obs=1,min_radius=min_radius,verbose=verbose,corr_aware=corr_pol,ag_start=ag_start,sim_r0=[source_x0,source_y0],isotropic=isotropic,save_beliefs=save_beliefs,ignore_errors=ignore_errors,deltas=save_deltas,save_displacements=save_displacements,force_source=force_source,exponents=exponents)
mean,err,failure=conditional_mean(times,tmax)
print('policy had mean arrival time',mean,'+/-',err,'with failure rate',failure)


data={
"times":times,
"sources":starts,
"hits":hits,
"bad_traj":bad_traj,
"entropies":entropies,
"actions":actions,
"beliefs":beliefs,
"deltas":deltas,
"dx":displacement_x,
"dy":displacement_y,
"positions":pos
}


with open(data_dir+'/'+data_file,'wb') as f:
    pickle.dump(data,f)


    
