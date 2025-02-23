from scipy.special import gamma as Gamma
import numpy as np
from scipy.stats import poisson as Poisson_distribution
from scipy.special import kv,kn
import random

def get_likelihood_from_conc(conc,threshold,tstep=1,isotropic=False):
    conc_arr=np.array(conc)
    thresholded=conc_arr>=threshold
    if isotropic:
        thresholded2=np.concatenate([thresholded,thresholded[:,:,:,::-1],np.transpose(thresholded[:,:,:,::-1],axes=(0,1,3,2)),np.transpose(thresholded[:,:,:,:],axes=(0,1,3,2)),thresholded[:,:,::-1,:],np.transpose(thresholded[:,:,::-1,:],axes=(0,1,3,2)),thresholded[:,:,::-1,::-1],np.transpose(thresholded[:,:,::-1,::-1],axes=(0,1,3,2))])
    else:
        thresholded2=np.concatenate([thresholded,thresholded[:,:,:,::-1]])
    l_un=np.sum(thresholded2,axis=(0,1))/(thresholded2.shape[0]*thresholded2.shape[1])
    l_un[np.isnan(l_un)]=0
    shifts = [(0,1),(1,0),(-1,0),(0,-1)]
    l0=[np.sum(thresholded2[:,1:,:,:]*np.roll(1-thresholded2[:,:-1,:,:],shift,axis=(2,3)),axis=(0,1),where=~np.roll(thresholded2[:,:-1,:,:],shift,axis=(2,3)))/np.sum(np.roll(1-thresholded2[:,:-1,:,:],shift,axis=(2,3)),axis=(0,1)) for shift in shifts]
    l1=[np.sum(thresholded2[:,1:,:,:]*np.roll(thresholded2[:,:-1,:,:],shift,axis=(2,3)),axis=(0,1),where=np.roll(thresholded2[:,:-1,:,:],shift,axis=(2,3)))/np.sum(np.roll(thresholded2[:,:-1,:,:],shift,axis=(2,3)),axis=(0,1)) for shift in shifts]
    for l in l0:
        l[np.isnan(l)]=0
    for l in l1:
        l[np.isnan(l)]=0
    return {'p_unconditional':l_un,'p_right_whiff':l1[1],'p_right_blank':l0[1],'p_left_whiff':l1[2],'p_left_blank':l0[2],'p_up_whiff':l1[0],'p_up_blank':l0[0],'p_down_whiff':l1[3],'p_down_blank':l1[3]}

def in_cone(env,r_ag,r0,tol=1e-3,n=1):
    return env.get_likelihood(r_ag[0]-r0[0],r_ag[1]-r0[1],1)>tol

def volume_ball(r, Ndim=2, norm='Euclidean'):
        if Ndim is None:
            Ndim = self.Ndim
        if norm == 'Manhattan':
            pm1 = 1
        elif norm == 'Euclidean':
            pm1 = 1 / 2
        elif norm == 'Chebyshev':
            pm1 = 0
        else:
            raise Exception("This norm is not implemented")
        return (2 * Gamma(pm1 + 1) * r) ** Ndim / Gamma(Ndim * pm1 + 1)

def initial_hit_aurore(env,hit=None):
    if hit is None:
        p_hit_table = np.zeros(env.numobs-1)
        r = np.arange(1, int(1000 * np.sqrt(env.D*env.tau)/env.dx))    
        shell_volume = volume_ball(r+0.5) - volume_ball(r-0.5)
        for h in range(1, env.numobs-1):
            p = Poisson(env,mean_number_of_hits(env,r), h)  # proba hit=h as a function of distance r to the source
            p_hit_table[h] = max(0, np.sum(p * shell_volume))  # not normalized
        p_hit_table /= np.sum(p_hit_table)
        hit = np.random.RandomState().choice(range(env.numobs-1), p=p_hit_table)
    return hit

def Poisson(env, mu, h):
        if h < env.numobs - 2:   # = Poisson(mu,hit=h)
            p = Poisson_unbounded(mu, h)
        elif h == env.numobs - 2:     # = Poisson(mu,hit>=h)
            sum = 0.0
            for k in range(h):
                sum += Poisson_unbounded(mu, k)
            p = 1 - sum
        else:
            raise Exception("h cannot be > Nhits - 1")
        return p


def Poisson_unbounded(mu, h):
        p = Poisson_distribution(mu).pmf(h)
        return p

def mean_number_of_hits(env, distance,Ndim=2):
        distance = np.array(distance)
        distance[distance == 0] = 1.0
        if Ndim == 1:
            mu = np.exp(-distance / np.sqrt(env.D*env.tau)*env.dx + 1)
        elif Ndim == 2:
            mu = kn(0, distance / np.sqrt(env.D*env.tau)*env.dx)/ kn(0, 1)
        elif Ndim == 3:
            mu = np.sqrt(env.D*env.tau)/env.dx / distance * np.exp(-distance / np.sqrt(env.D*env.tau)*env.dx + 1)
        elif Ndim > 3:
            mu = (np.sqrt(env.D*env.tau)/env.dx / distance) ** (Ndim / 2 - 1) \
                 * kv(Ndim / 2 - 1, distance*env.dx/ np.sqrt(env.D*env.tau)) \
                 / kv(Ndim / 2 - 1, 1)
        else:
            raise Exception("Problem with the number of dimensions")
        mu *= mu0_Poisson(env)
        return mu

def mu0_Poisson(env,Ndim=2):
        """Sets the value of mu0_Poisson (mean number of hits at distance = lambda), which is derived from the
         physical dimensionless parameters of the problem. It is required by _mean_number_of_hits().
        """
        dx_over_a = env.dx/env.agent_size  # agent step size / agent radius
        lambda_over_a = np.sqrt(env.D*env.tau)/env.agent_size
        a_over_lambda = 1.0 / lambda_over_a

        if Ndim == 1:
            mu0_Poisson = 1 / (1 - a_over_lambda) * np.exp(-1)
        elif Ndim == 2:
            mu0_Poisson = 1 / np.log(lambda_over_a) * kn(0, 1)
        elif Ndim == 3:
            mu0_Poisson = a_over_lambda * np.exp(-1)
        elif Ndim > 3:
            mu0_Poisson = (Ndim - 2) / Gamma(Ndim / 2) / (2 ** (Ndim / 2 - 1)) * \
                          a_over_lambda ** (Ndim - 2) * kv(Ndim / 2 - 1, 1)
        else:
            raise Exception("problem with Ndim")

        mu0_Poisson *= env.R*env.dt
        return mu0_Poisson

def get_data_from_file(file_list,good_starts,tmax=2500):
    indices=random.choice(good_starts)
    with open(file_list[indices[0]],'rb') as f:
        data=pickle.load(f)
    if random.random()<0.5:
        conc=data['concentration'][:,:,::-1]
        tstart=indices[1]
        return conc[tstart:tstart+tmax,...],indices[2],conc.shape[2]-indices[3]-1,indices[0],tstart
    conc=data['concentration']
    tstart=indices[1]
    return conc[tstart:tstart+tmax,...],indices[2],indices[3],indices[0],tstart

def get_random_data(conc_list,good_starts,tmax=2500):
    indices=random.choice(good_starts)

    data=conc_list[indices[0]]
    if random.random()<0.5:
        conc=data[:,:,::-1]
        tstart=indices[1]
        return conc[tstart:tstart+tmax,...],indices[2],conc.shape[2]-indices[3]-1,indices[0],tstart
    conc=data
    tstart=indices[1]
    return conc[tstart:tstart+tmax,...],indices[2],indices[3],indices[0],tstart

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
