import numpy as np
import random
import timeit
import sys
import pickle
from copy import deepcopy
from numba import njit,prange,types,config
from multiprocessing import Pool
from functools import partial
from operator import add
#from multiprocessing import Pool
from timeit import default_timer as timer
#np.set_printoptions(threshold=sys.maxsize)
config.THREADING_LAYER='omp'

class belief:
    def __init__(self,b,dist=None):
        self.data=b
        self.alpha=None
        self.value=None
        self.dist=None
        self.tau=None
        self.p_obs=None
        self.error=None
        self.cluster_probs=None

class alpha_vec:
    def __init__(self,alph,a,g=None):
        self.data=alph
        self.gs=g
        self.action=a
        self.used=0

    def __eq__(self, other):
         return np.array_equal(self.data,other.data)


class ValueFunction:
    def __init__(self,env,beliefs=None,distances=None,v0=None,a0=None,ordered=False,shaped=False):
        self.env=env
        shape=env.rewards[0].shape
        self.shape=shape
        self.alphas=None
        self.beliefs=[]
        self.maxb=None
        self.ordered=ordered
        self.shaped=shaped

        if v0!=None:
            alpha0=1/(1-env.gamma)*v0*np.ones(shape)
            gs=[]
            for a in env.actions:
                gs.append(env.get_g(alpha0,a))
            newalpha=alpha_vec(alpha0,a0,gs)
            self.alphas=[newalpha]
            self.alphaarray=np.expand_dims(newalpha.data,0)

        self.maxvalue=-np.inf
        if beliefs!=None:
            for b in beliefs:
                newb=belief(b)
                v,alpha=self.value(b.data)
                if v > self.maxvalue:
                    self.maxvalue=v
                    self.maxb=b
                newb.value=v
                newb.alpha=alpha
                self.beliefs.append(newb)
            if self.ordered:
                i=0
                for b in beliefs:
                    b.dist=distances[i]
                    i+=1
                self.beliefs.sort(key=lambda x: x.dist)
            self.barray=np.zeros((len(self.beliefs),)+self.shape)
            i=0
            for b in self.beliefs:
               self.barray[i,...]=b.data
               i+=1


        if beliefs==None:
            self.beliefs=[]
        self.maxvalue_history=[self.maxvalue]
        self.changed_beliefs=[]
        self.iterations=0
        if self.shaped:
            self.shaping=self.env.shaping
        else:
            self.shaping=np.zeros((self.env.numactions,))


    def compute_taus_probs(self):
        for b in self.beliefs:
            if b.tau is None:
                b.tau=[]
                b.p_obs=[]
                for a in self.env.actions:
                    p,tau=self.env.transition_function(b.data,a)
                    newtau=[]
                    for t in tau:
                        newtau.append(belief(t))
                    b.tau.append(newtau)
                    b.p_obs.append(p)

    def value(self,b,alphaset=None,parallel=False,softmax=False):
        if alphaset==None:
            alphaset=self.alphas
        if softmax:
            value,index=einsum_value_softmax(b.data,self.alphaarray,self.env.gamma)
            alpha=self.alphas[int(index)]
            return value,alpha
        if parallel:
            #value,index=parallel_value_numba_single_belief(np.broadcast_to(b.data,(self.alphaarray.shape[0],)+b.data.shape),self.alphaarray)
            value,index=einsum_value_single_belief(b.data,self.alphaarray)
            alpha=self.alphas[int(index)]
            return value,alpha

        else:
            out=-np.inf
            bestalpha=None
            for alpha in alphaset:
                tmp=np.sum(b*alpha.data)
                if tmp>out:
                    out=tmp
                    bestalpha=alpha
            if bestalpha is None:
                raise RuntimeError('best alpha is None')
            return out,bestalpha

    def initiate_clusters(self,maxdist=15):
        #self.maxdist=self.beliefs[-1].dist
        print("initiating clusters")
        self.maxdist=maxdist
        i=0
        for b in self.beliefs:
            if i%100==0:
                print("belief ",i)
            b.cluster_probs=[]
            for d in range(1,maxdist+1):
                p=0
                for i in range(-(self.env.dims[0]-1),self.env.dims[0]):
                    for j in range(-(self.env.dims[1]-1),self.env.dims[1]):
                        if np.abs(i)+np.abs(j)==d:
                            p+=b.data[i,j]
                b.cluster_probs.append(p)
            i+=1


    def bellmanError(self,b):
        beste=-np.inf
        for a in range(self.env.numactions):
            e=0
            for obs in range(self.env.numobs):
                v,poop=self.value(b.tau[a][obs],parallel=True)
                if np.isinf(v):
                    raise RuntimeError("infinite value encountered for a tau")
                e+=self.env.gamma*b.p_obs[a][obs]*v
                if np.isinf(e):
                    raise RuntimeError("infinite p_obs encountered")
            e+=np.sum(b.data*(self.env.rewards[a]))
            if np.isinf(e):
                raise RuntimeError("expected reward is infinite")
            if e>beste:
                beste=e
        if np.isinf(beste):
            raise RuntimeError("infinite bellman error RHS encountered")
        if np.isinf(b.value):
            raise RuntimeError("belief has infinite value")       
        return beste-b.value

    def bellmanErrorNew(self,b,value=None):
        newb=belief(b)
        beste=-np.inf
        taus=[]
        p_obs=[]
        for a in self.env.actions:
            p,tau=self.env.transition_function(newb.data,a)
            newtau=[]
            for t in tau:
                newtau.append(belief(t))
            taus.append(newtau)
            p_obs.append(p)
        for a in range(self.env.numactions):
            e=0
            for obs in range(self.env.numobs):
                v,poop=self.value(taus[a][obs],parallel=True)
                e+=self.env.gamma*p_obs[a][obs]*v
            e+=np.sum(newb.data*(self.env.rewards[a]))
            if e>beste:
                beste=e
        if value is None:
            v,poop=self.value(newb,parallel=True)
        else:
            v=value
        return beste-v

    def iterateV(self,parallel=False,verbose=False,v_tol=1e-12,prioritized=False,clusters=False,min_improve=0,boltzmann=False,v_frac_tol=0,pvi=False,max_frac_tol=np.inf):
        errors=[]
        mean_bellman=None
        max_bellman=None
        error_start=timer()
        if True:
            for b in self.beliefs:
                b.error=self.bellmanError(b)
                errors.append(b.error)
            mean_bellman=np.sum(np.abs(errors))/len(errors)
            max_bellman=max(np.abs(errors))
            rms_bellman=np.sqrt(np.mean(np.array(errors)**2))
            print("maximum bellman error:",max_bellman)
            print("mean bellman error:",mean_bellman)
            print("rms bellman error:",rms_bellman)
        error_end=timer()
        error_time=error_end-error_start


        Bprime=self.beliefs.copy()
        if pvi:
            probs=[x.error for x in self.beliefs]
            probs=np.exp(probs)
            probs=probs/np.sum(probs)
            b=np.random.choice(self.beliefs,p=probs)
            alpha=self.backup(b)
            gs=[]
            for act in self.env.actions:
                gs.append(self.env.get_g(alpha.data,act)) #TODO: turn into NP array???
            alpha.gs=gs
            self.alphas.append(alpha)
            Bprime=[]

        if prioritized and not boltzmann:
            Bprime.sort(key=lambda x: x.error,reverse=True)

        if clusters:
            if self.currdist is None:
                self.currdist=0
            self.currdist+=1
            Bprime.sort(key=lambda x: x.cluster_probs[self.currdist-1])

        newalpha=[]

        Nb=0
        backedup=[]
        backup_time=0
        bprime_time=0
        while(Bprime):
            if verbose:
                print("B prime has ",len(Bprime)," elements")
            if self.ordered or (prioritized and not boltzmann):
                b=Bprime[0]
                #print(b.error)
                #print(b.dist)
            elif prioritized and boltzmann:
                errorz=[x.error for x in Bprime]
                temp=1
                probs=np.exp(np.array(errorz)/temp)
                probs=probs/np.sum(probs)
                try:
                    b=np.random.choice(Bprime,p=probs)
                except:
                    print('errors are',errorz)
                    print('probabilities are',probs)
                    raise RuntimeError('problem computing prioritization probabilities')
            else:
                b=random.choice(Bprime)
            backup_1=timer()
            alpha=self.backup(b)
            backup_2=timer()
            backup_time+=backup_2-backup_1
            val=b.value
            curralpha=b.alpha
            if b.alpha.gs is None:
                raise RunTimeError("this belief's alpha has no computed g vectors")
            newval=np.sum(b.data*alpha.data)

            #if verbose:
            #    print("next belief has entropy",-np.sum(b.data*np.log(b.data,out=np.zeros_like(b.data),where=b.data!=0)),"and mean distance",np.sum(b.data*self.env.dist))
            if val!=0 and newval>val*(1+max_frac_tol):
                if verbose:
                    print("backup improved value TOO MUCH, let's shut it down")
                alpha=curralpha
            elif newval>val*(1+min_improve):
                if verbose:
                    print("backup improved value sufficiently")
                gs=[]
                for act in self.env.actions:
                    gs.append(self.env.get_g(alpha.data,act)) #TODO: turn into NP array???
                alpha.gs=gs
                backedup.append(b)
            else:
                if verbose:
                    print("backup didn't improve value sufficiently")
                alpha=curralpha

            newalpha.append(alpha)
            bprime_1=timer()
            Bprime = [bee for bee in Bprime if np.sum(bee.data*alpha.data)*(1+v_frac_tol)+v_tol<bee.value]
            bprime_2=timer()
            bprime_time+=bprime_2-bprime_1
        # if self.ordered:
        #     Bprime.sort(key=lambda x: x.dist)
        if not pvi:
            self.alphas=newalpha
        newmax=-np.inf
        newmaxb=None
        
        if not parallel:
            self.alphaarray=np.zeros((len(self.alphas),)+self.shape)
            i=0
            for alpha in self.alphas:
                self.alphaarray[i,...]=alpha.data
                i+=1
            values,indices=einsum_value(self.barray,self.alphaarray)
            i=0
            for boo in self.beliefs:
                oldalpha=boo.alpha
                v=values[i]
                index=indices[i]
                alf=self.alphas[index]
                #assert np.array_equal(alf.data,np.squeeze(alphaarray[index,:,:]))
                if boo.value>v*(1+v_frac_tol)+v_tol:
                    raise RuntimeError("value of at least one belief decreased! old value was ",boo.value," new value is ",v)
                boo.value=v
                if v>newmax:
                    newmax=v
                    newmaxb=boo
                boo.alpha=alf
                if not np.array_equal(alf.action,oldalpha.action):
                    Nb+=1
                i+=1
        elif False:
            alphaarray=np.zeros((len(self.alphas),)+self.shape)
            i=0
            for alpha in self.alphas:
                alphaarray[i,:,:]=alpha.data
                i+=1
            with Pool() as pool:
                values_and_indices=pool.map(partial(value_for_pool,alphas=alphaarray),self.beliefs)
            newbeliefs=[]
            for item in value_and_indices:
                boo=item[0]
                oldalpha=boo.alpha.copy()
                v=item[1]
                index=item[2]
                if v<boo.value:
                    raise RuntimeError("value of at least one belief decreased! old value was ",boo.value," new value is ",v)
                boo.value=v
                if v>newmax:
                    newmax=v
                    newmaxb=boo
                boo.alpha=self.alphas[index]
                if not np.array_equal(boo.alpha.data,oldalpha.data):
                    Nb+=1
                newbeliefs.append(boo)
            assert len(self.beliefs)==len(newbeliefs)
            self.beliefs=newbeliefs

        else:
            ### parallel value accumulation using njit ###
            array_1=timer()
            self.alphaarray=np.zeros((len(self.alphas),)+self.shape)
            i=0
            for alpha in self.alphas:
                self.alphaarray[i,...]=alpha.data
                i+=1
            array_2=timer()
            array_time=array_2-array_1
            v_update_1=timer()
            if len(self.barray.shape)==4:
                values,indices=parallel_value_numba_4(self.barray,self.alphaarray)
            else:
                values,indices=parallel_value_numba(self.barray,self.alphaarray)
            i=0
            for boo in self.beliefs:
                oldalpha=boo.alpha
                v=values[i]
                index=indices[i]
                alf=self.alphas[int(index)]

                if v<boo.value-v_tol:
                    raise RuntimeError("value of at least one belief decreased! old value was ",boo.value," new value is ",v)
                if v>newmax:
                    newmax=v
                    newmaxb=boo
                boo.value=v
                boo.alpha=alf
                if not np.array_equal(alf.action,oldalpha.action):
                    Nb+=1
                i+=1
            v_update_2=timer()
            v_update_time=v_update_2-v_update_1


        oldmax=self.maxvalue
        # if oldmax>newmax:
        #     if np.array_equal(self.maxb.data,newmaxb):
        #         raise RuntimeError("max value got worse! old max was ",oldmax, ". new max is ",newmax,". associated beliefs are the SAME")
        #     else:
        #         raise RuntimeError("max value got worse! old max was ",oldmax, ". new max is ",newmax,". associated beliefs are DIFFERENT. new value of old max belief is ", self.maxb.value)
        self.maxvalue=newmax
        self.maxb=newmaxb

        self.iterations+=1
        self.maxvalue_history.append(self.maxvalue)
        self.changed_beliefs.append(Nb)
        return newmax-oldmax,Nb,backedup,max_bellman,mean_bellman,rms_bellman,error_time,backup_time,bprime_time,array_time,v_update_time

    def backup(self,b):
        tmp1=-np.inf
        bestg=None
        besta=None
        for a in range(self.env.numactions):
            gb=self.env.rewards[a].copy()
            #print(np.sum(gb))
            sumvec=0
            for obs in range(self.env.numobs):
                tmp2=-np.inf
                gopt=None
                for alpha in self.alphas:
                    gi=alpha.gs
                    tmp3=np.sum(b.data*gi[a][obs])
                    if tmp3>tmp2:
                        tmp2=tmp3
                        gopt=gi[a][obs].copy()
                sumvec=sumvec+gopt
            gb+=self.env.gamma*sumvec
            #print(np.sum(gb))

            tmp4=np.sum(b.data*gb)
            if tmp4>tmp1:
                tmp1=tmp4
                bestg=gb
                besta=a
        out=alpha_vec(bestg,self.env.actions[besta],None)
        return out

    def computeOptimal(self,tol=0.001,maxit=None,minit=1,criterion="maxv",minalpha=None,parallel=False,verbose=False,v_tol=1e-12,prioritized=False,clusters=False,min_improve=0,boltzmann=False,v_frac_tol=0,pvi=False,max_frac_tol=np.inf):
        stage=0
        while True:
            print("***********************************")
            print("BACKUP STAGE ",self.iterations)
            print("***********************************")

            print("max value: ",self.maxvalue)
            change,Nb,backedup,max_bellman,mean_bellman,rms_bellman,error_time,backup_time,bprime_time,array_time,v_update_time=self.iterateV(parallel=parallel,verbose=verbose,v_tol=v_tol,prioritized=prioritized,clusters=clusters,min_improve=min_improve,boltzmann=boltzmann,v_frac_tol=v_frac_tol,pvi=pvi,max_frac_tol=max_frac_tol)
            #print("change in max value: ",change)
            stage+=1
            print(Nb," beliefs' optimal actions changed")
            #if Nb==0 or Nb==len(self.beliefs):
            #    i+=1
            #    continue
            if stage<minit:
                continue
            if maxit!=None and stage>=maxit:
                break

            if minalpha is not None:
                if len(self.alphas)<minalpha:
                    continue
            if criterion=="maxv":
                if np.abs(change/self.maxvalue)<tol:
                    break
            if criterion=="b":
                if Nb< tol:
                    break
        return backedup,max_bellman,mean_bellman,rms_bellman,error_time,backup_time,bprime_time,array_time,v_update_time

    def prune_unused_alphas(self,threshold=0,fraction=0):
        tally=0
        initial=len(self.alphas)
        for alpha in self.alphas:
            if alpha.used<threshold:
                self.alphas.remove(alpha)
                tally+=1
                del(alpha)
        self.alphas.sort(key=lambda x: x.used)
        for alpha in self.alphas:
            if initial*(1-fraction)>=len(self.alphas):
                break
            self.alphas.remove(alpha)
            tally+=1
            del(alpha)
        for alpha in self.alphas:
            alpha.used=0
        self.alphaarray=np.zeros((len(self.alphas),)+self.shape)
        i=0
        for alpha in self.alphas:
            self.alphaarray[i,...]=alpha.data
            i+=1
        values,indices=parallel_value_numba(self.barray,self.alphaarray)
        i=0
        newmax=-np.inf
        for boo in self.beliefs:
            v=values[i]
            index=indices[i]
            alf=self.alphas[int(index)]
            boo.value=v
            if v>newmax:
                newmax=v
                newmaxb=boo
            i+=1
        self.maxvalue=newmax
        self.maxb=newmaxb
        print("removed ",tally," underused alphas")

    def save(self,filename,no_beliefs=False):
        vf=deepcopy(self)
        if no_beliefs:
            vf.beliefs=None
            vf.barray=None

        with open(filename + '.pkl', 'wb') as f:
            pickle.dump(vf, f, pickle.HIGHEST_PROTOCOL)

    def load(self,filename):
        with open(filename + '.pkl', 'rb') as f:
            vf=pickle.load(f)
            self.alphas=vf.alphas
            self.beliefs=vf.beliefs
            self.maxvalue=vf.maxvalue
            self.iterations=vf.iterations
            self.maxvalue_history=vf.maxvalue_history
            self.changed_beliefs=vf.changed_beliefs
            self.barray=vf.barray
            self.alphaarray=vf.alphaarray

    def add_beliefs(self,beliefs,distances=None):
        i=0
        if self.beliefs is None:
            self.beliefs=[]
        for b in beliefs:
            bel=belief(b)
            v,alpha=self.value(b)
            bel.value=v
            bel.alpha=alpha
            if self.ordered:
                bel.dist=distances[i]
            i+=1
            if v > self.maxvalue:
                self.maxvalue=v
            self.beliefs.append(bel)
        if self.ordered:
            self.beliefs.sort(key=lambda x: x.dist)
        self.barray=np.zeros((len(self.beliefs),)+self.shape)
        i=0
        for b in self.beliefs:
           self.barray[i,...]=b.data
           i+=1

    def shift_value(self,alpha):
        self.alphaarray+=alpha
        g=[self.env.get_g(alpha,action) for action in self.env.actions]
        for a in self.alphas:
            a.data=a.data+alpha
            for act in range(self.env.numactions):
                for obs in range(self.env.numobs):
                    a.gs[act][obs]+=g[act][obs]
        for b in self.beliefs:
            b.value=b.value+np.sum(b.data*alpha)
    
    def load_alphas(self,alphas):
        self.alphas=alphas
        self.alphaarray=np.zeros((len(self.alphas),)+self.shape)
        i=0
        for alpha in self.alphas:
            self.alphaarray[i,...]=alpha.data
            i+=1


def check_b(b,alpha,beliefdict):
    if np.sum(b*alpha)<beliefdict[b.tobytes()][0]:
        return b
#
@njit(parallel=True)
def get_best_alphas(beliefs,alphas):
    size=beliefs.shape[0]
    dims=beliefs[0,:,:].shape
    bestalphas=np.empty(beliefs.shape)
    for i in prange(size):
        b=beliefs[i,:,:] # this will not work if beliefs are not 2D
        value=-np.inf
        bestalpha=np.empty(dims)
        for j in range(alphas.shape[0]):
            alpha=alphas[j,:,:]
            tmp=np.sum(b*alpha)
            if tmp>value:
                value=tmp
                bestalpha=alpha
        bestalphas[i,:,:]=bestalpha
    return bestalphas
#
@njit(parallel=True)
def parallel_value_numba(beliefs,alphas):
    values  = np.empty(beliefs.shape[0])
    indices = np.empty(beliefs.shape[0])
    for i in prange(beliefs.shape[0]):
        dot = np.zeros(alphas.shape[0])
        for j in prange(alphas.shape[0]):
            for k in prange(alphas.shape[1]):
                for l in prange(alphas.shape[2]):
                    dot[j] += beliefs[i,k,l]*alphas[j, k, l]
        index=np.argmax(dot)
        values[i]=dot[index]
        indices[i]=index
    return values,indices

@njit(parallel=True)
def parallel_value_numba_4(beliefs,alphas):
    values  = np.empty(beliefs.shape[0])
    indices = np.empty(beliefs.shape[0])
    for i in prange(beliefs.shape[0]):
        dot = np.zeros(alphas.shape[0])
        for j in prange(alphas.shape[0]):
            for k in prange(alphas.shape[1]):
                for l in prange(alphas.shape[2]):
                    for m in prange(alphas.shape[3]):
                        dot[j] += beliefs[i,k,l,m]*alphas[j, k, l, m]
        index=np.argmax(dot)
        values[i]=dot[index]
        indices[i]=index
    return values,indices

@njit(parallel=True)
def parallel_value_numba_single_belief(belief,alphas):
    product=np.multiply(belief,alphas)
    dot1=np.sum(product,axis=2)
    dot2=np.sum(dot1,axis=1)
    index=np.argmax(dot2)
    value=dot2[index]
    return value,index

def einsum_value_single_belief(belief,alphas):
    if len(belief.shape)==3:
        dot=np.einsum('klm,jklm->j',belief,alphas)
        index=np.argmax(dot)
        value=dot[index]
        return value,index
    dot=np.einsum('kl,jkl->j',belief,alphas)
    index=np.argmax(dot)
    value=dot[index]
    return value,index

def einsum_value_softmax(belief,alphas,gamma):
    dot=np.einsum('kl,jkl->j',belief,alphas)
    probs=np.exp(dot/(1-gamma)/np.max(dot))
    probs=probs/np.sum(probs)
    index=np.random.choice(range(len(probs)),p=probs)
    value=dot[index]
    return value,index

def einsum_value(beliefs,alphas):
    if len(beliefs.shape==4):
        dot=np.einsum('iklm,jklm->ij',beliefs,alphas)
        indices=np.argmax(dot,axis=1)
        values = dot[np.arange(len(indices)), indices]
        return values,indices
    dot=np.einsum('ikl,jkl->ij',beliefs,alphas)
    indices=np.argmax(dot,axis=1)
    values = dot[np.arange(len(indices)), indices]
    #assert np.array_equal(values,np.max(dot,axis=1))
    return values,indices

def value_for_pool(b,alphas):
    axes=tuple(range(1,alphas.ndim))
    dot=np.sum(b.data*alphas,axis=axes)
    index=np.argmax(dot)
    return b,dot[index],index
