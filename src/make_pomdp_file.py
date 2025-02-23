import sys
import numpy as np
import environment
import agent
import pickle
import utils
import os


def transition(state,action,env):
    pos=np.asarray(np.unravel_index(state,dims))
    if np.array_equal(pos,[0,0]):
        return state
    if action=="north":
        a=np.array([0,1])
    if action=="south":
        a=np.array([0,-1])
    if action=="east":
        a=np.array([1,0])
    if action=="west":
        a=np.array([-1,0])
    new_pos=pos+a

    if new_pos[0]<-(env.dims[0]-1):
        new_pos[0]+=2*env.dims[0]-1
    elif new_pos[0]>(env.dims[0]-1):
        new_pos[0]-=2*env.dims[0]-1
    if new_pos[1]<-(env.dims[1]-1):
        new_pos[1]+=2*env.dims[1]-1
    elif new_pos[1]>(env.dims[1]-1):
        new_pos[1]-=2*env.dims[1]-1
    return indices[tuple(new_pos)]


def reward(state,action,env):
    if state==0:
        return 0
    rew_matrix=env.rewards
    if action=="east":
        tmp=rew_matrix[0].flatten()
    elif action=="west":
        tmp=rew_matrix[1].flatten()
    elif action=="north":
        tmp=rew_matrix[2].flatten()
    elif action=="south":
        tmp=rew_matrix[3].flatten()
    return tmp[state]

max_detections=1
if 'GAMMA' in os.environ:
    gamma=float(os.environ.get('GAMMA'))
else:
    gamma=0.98

import os

shape_x = int(os.environ.get('SHAPE_X'))
shape_y = int(os.environ.get('SHAPE_Y'))

source_x=int(os.environ.get('SOURCE_X0'))
source_y=int(os.environ.get('SOURCE_Y0'))

if 'THRESHOLD' in os.environ:
    threshold=float(os.environ.get('THRESHOLD'))
else:
    threshold=None

env_params={
  "data":None,
  "threshold":threshold,
}

if 'CORR_POL' in os.environ:
    corr=bool(int(os.environ.get('CORR_POL')))
else:
    corr=False

sim_r0=(source_x,source_y)

dims=(shape_x,shape_y)

env=environment.OlfactorySearch2D(dims,dummy=False,**env_params,corr=True)

#with open('/mnt/beegfs/heinonen/v9_prob_0.1_percent.pkl','rb') as f:
#    l_un=pickle.load(f)
#l0=None
#l1=None

if 'ISOTROPIC' in os.environ:
    isotropic=bool(int(os.environ.get('ISOTROPIC')))
else:
    isotropic= False

if 'LIKELIHOOD_FILE' in os.environ:
    likelihood_file=os.environ.get('LIKELIHOOD_FILE')
    likelihood_dir=os.environ.get('LIKELIHOOD_DIR')
    with open(likelihood_dir+'/'+likelihood_file,'rb') as f:
        ells=pickle.load(f)
else:
    conc_file=os.environ.get('CONC_FILE')
    conc_dir=os.environ.get('CONC_DIR')
    if 'TSTEP' in os.environ:
        tstep=float(os.environ.get('TSTEP'))
    else:
        tstep=1

    with open(conc_dir+'/'+conc_file,'rb') as f:
        conc=pickle.load(f)
    ells=utils.get_likelihood_from_conc(conc,threshold=threshold,isotropic=isotropic,tstep=tstep)

l0=[ells['p_right_blank'],ells['p_left_blank'],ells['p_up_blank'],ells['p_down_blank']]
l1=[ells['p_right_whiff'],ells['p_left_whiff'],ells['p_up_whiff'],ells['p_down_whiff']]
l_un=ells['p_unconditional']
env.set_likelihood(l_un,l0,l1,sim_r0=sim_r0)


global indices

dims=(2*shape_x-1,2*shape_y-1)
n_states=dims[0]*dims[1]

ag_start_x=int(os.environ.get('AG_START_X'))
ag_start_y=int(os.environ.get('AG_START_Y'))

likelihoods=[]
for i in range(2):
    l=[]
    for j in range(4):
        l.append(np.roll(env.likelihood[i][j],(env.dims[0],env.dims[1]),axis=(0,1)).flatten())
    likelihoods.append(l.copy())
ag=agent.CorrAgent(env,np.array([ag_start_x,ag_start_y]))
l_un_flattened=np.roll(env.unconditional_likelihood,(env.dims[0],env.dims[1]),axis=(0,1)).flatten()

indices=np.reshape(np.arange(n_states),dims)
actions=['east','west','north','south']

ag.updateBelief(1,None)
initial_belief=ag.perseus_belief(ag.belief)
initial_belief/=np.sum(initial_belief)
initial_belief=initial_belief.flatten()

pomdp_file=os.environ.get('POMDP_FILE')
pomdp_dir=os.environ.get('POMDP_DIR')
with open(pomdp_dir+'/'+pomdp_file,'w') as f:
    f.write('discount: ')
    f.write(str(gamma))
    f.write('\n')
    f.write('values: reward\n')
    f.write('states: ')
    if corr:
        f.write(str(2*n_states))
    else:
        f.write(str(n_states))
    f.write('\n')
    f.write('actions: east west north south\n')
    f.write('observations: '+str(max_detections+2)+'\n')
    f.write('start:\n')
    if corr:
        for state in range(n_states):
            f.write('0 ')
    for state in range(n_states):
        f.write(str(initial_belief[state])+' ')
    f.write('\n')
    for initial in range(n_states):
        for i,a in enumerate(actions):
            final=transition(initial,a,env)
            if final==0:
                f.write('T: '+a+' : '+str(initial)+' : '+str(final)+' '+'1'+"\n")
                if corr:
                    f.write('T: '+a+' : '+str(initial+n_states)+' : '+str(final+n_states)+' 1\n')
            elif initial==0 or initial==n_states:
                f.write('T: '+a+' : '+str(initial)+' : '+str(initial)+' 1\n')
            elif corr:
                #probability of no detection given no detection
                f.write('T: '+a+' : '+str(initial)+' : '+str(final)+' '+str(1-likelihoods[0][i][final])+"\n")
                #probability of no detection given detection
                f.write('T: '+a+' : '+str(initial+n_states)+' : '+str(final)+' '+str(1-likelihoods[1][i][final])+"\n")
                #probability of detection given no detection
                f.write('T: '+a+' : '+str(initial)+' : '+str(final+n_states)+' '+str(likelihoods[0][i][final])+"\n")
                #probability of detection given detection
                f.write('T: '+a+' : '+str(initial+n_states)+' : '+str(final+n_states)+' '+str(likelihoods[1][i][final])+"\n")
            else:
                f.write('T: '+a+' : '+str(initial)+' : '+str(final)+' 1 \n')
    for final in range(n_states):
        f.write('O: * : '+str(final)+'\n')
        if final==0:
            for i in range(max_detections+1):
                f.write(str(0)+' ')
            f.write(str(1)+'\n')
        elif corr:
            f.write('1 0 0 \n')
        else:
            f.write(str(1-l_un_flattened[final])+' '+str(l_un_flattened[final])+' 0\n')
    if corr:
        for final in range(n_states,2*n_states):
            f.write('O: * : '+str(final)+'\n')
            if final==n_states:
                for i in range(max_detections+1):
                    f.write(str(0)+' ')
                f.write(str(1)+'\n')
            else:
                f.write('0 1 0 \n')


    for state in range(n_states):
        for a in actions:
            r=reward(state,a,env)
            if not r==0:
                f.write('R: '+a+' : '+str(state)+' : * : * '+str(r)+'\n')
                if corr:
                    f.write('R: '+a+' : '+str(state+n_states)+' : * : * '+str(r)+'\n')
