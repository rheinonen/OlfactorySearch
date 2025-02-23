import random
import numpy as np
from datetime import datetime
from scipy.stats import entropy
import math
from timeit import default_timer as timer


class CastAndSurge:
    def __init__(self,agent):
        self.agent=agent
        self.tmax=0
        self.tau=0
        self.vertical_dir=1
    def getAction(self,belief):
        if self.agent.last_obs:
            self.tmax=0
            self.tau=0
            return np.array([-1,0])
        if self.tau<self.tmax:
            if self.vertical_dir==1:
                self.tau+=1
                return np.array([0,1])
            else:
                self.tau+=1
                return np.array([0,-1])
        if self.tau==self.tmax:
            self.tmax+=1
            self.tau=0
            self.vertical_dir=self.vertical_dir*(-1)
            return np.array([-1,0])


class RandomPolicy:
    def __init__(self,env):
       # self.seed=datetime.now()
       # random.seed(self.seed)
        self.actions=env.actions

    def getAction(self,belief):
        return random.choice(self.actions)

class ThompsonSampling:
    def __init__(self,agent,persistence_time=1):
        self.agent=agent
        self.persistence_time=persistence_time
        self.t=0
        self.location=None
    def reset(self):
        self.t=0
        self.location=None
    def getAction(self,belief):
        #print('heading towards',self.location)
        if np.array_equal(self.location,self.agent.true_pos):
            self.t=0
        if self.t==0:
            b=belief.flatten()
            indices=np.arange(b.shape[0])
            index=np.random.choice(indices,p=b)
            self.location=np.unravel_index(index,belief.shape)
        self.t+=1
        if self.t==self.persistence_time:
            self.t=0
        return random.choice(follow_loc(self.location,self.agent.true_pos))

class QMDPPolicy:
    def __init__(self,agent,gamma=None):
        self.agent=agent
        if gamma is not None:
            self.q=gamma**(self.agent.env.dist)
            self.q[0,0]=0
        else:
            self.q=self.agent.env.qvalue

    def getAction(self,belief):
        b=self.agent.perseus_belief(belief)
        bestaction=None
        best_r=-np.inf
        for a in self.agent.env.actions:
            q=np.roll(self.q,tuple(-a),axis=(0,1))
            if np.array_equal(a,[1,0]):
                q[self.agent.env.dims[0],:]=0
            if np.array_equal(a,[-1,0]):
                q[self.agent.env.dims[0]-1,:]=0
            if np.array_equal(a,[0,1]):
                q[:,self.agent.env.dims[1]-1]=0
            if np.array_equal(a,[0,-1]):
                q[:,self.agent.env.dims[1]]=0
            r=np.sum(b*q)
            #print(a,r)
            if r>best_r:
                best_r=r
                bestaction=a
        return bestaction

class TrivialPolicy:
    def __init__(self,action=np.array([0,0])):
        self.action=action
    def getAction(self,belief):
        return self.action

class ActionVoting:
    def __init__(self,agent):
        self.agent=agent
        dims=self.agent.env.dims

        self.delta_mat=np.zeros((self.agent.env.numactions,2*dims[0]-1,2*dims[1]-1))
        for i in range(-(dims[0]-1),dims[0]):
            for j in range(-(dims[1]-1),dims[1]):
                actions=follow_loc([0,0],[i,j])
                for action in range(self.agent.env.numactions):
                    if any((self.agent.env.actions[action] == x).all() for x in actions):
                        self.delta_mat[action,i,j]=1

    def getAction(self,belief):
        b=self.agent.perseus_belief(belief)
        bestaction=None
        best_p=-np.inf
        for a in range(self.agent.env.numactions):
            p=np.sum(self.delta_mat[a,:,:]*b)
            if p>best_p:
                best_p=p
                bestaction=a
        return self.agent.env.actions[bestaction]

class LocalGradientAscent:
    def __init__(self,agent,verbose=False):
        self.agent=agent
        self.verbose=verbose
    def getAction(self,belief):
        best_action=[]
        best_b=0
        pos=[]
        for a in self.agent.env.actions:
            newpos=self.agent.belief_env.transition(self.agent.true_pos,a)
            pos.append((newpos,a))
            b=self.agent.belief[newpos[0],newpos[1]]
            if b==best_b:
                best_action.append(a)
            elif b>best_b:
                best_action=[a]
                best_b=b
        depth=1
        while len(best_action)==len(self.agent.env.actions):
            if self.verbose:
                print('no best action at depth',depth)
            newpos=self.nextPos(pos,depth)
            best_action=[]
            best_b=0
            for p in newpos:
                b=self.agent.belief[p[0][0],p[0][1]]
                if self.verbose:
                    print('checking pos',p[0],'associated with first action',p[1])
                    print('b=',b)
                if b==best_b:
                    best_action.append(p[1])
                elif b>best_b:
                    best_b=b
                    best_action=[p[1]]
            depth+=1
        return random.choice(best_action)
      
    def nextPos(self,starts,depth):
        assert(depth>=1)
        pos=[]
        if depth==1:
            for start in starts:
                for a in self.agent.env.actions:
                    pos.append((self.agent.belief_env.transition(start[0],a),start[1]))
            return pos
        else:
            for start in starts:
                for a in self.agent.env.actions:
                    pos.append((self.agent.belief_env.transition(start[0],a),start[1]))
            return self.nextPos(pos,depth-1)
        

class OptimalPolicy:
    def __init__(self,vf,agent,parallel=False,epsilon=0):
        self.vf=vf
        self.agent=agent
        self.parallel=parallel
        self.set_used=False
        self.epsilon=epsilon
        self.last_value=None
    def getAction(self,belief):
        if random.random()<self.epsilon:
            return random.choice(self.agent.env.actions)
        b=self.agent.perseus_belief(belief) # this line for situations where belief used by perseus is defined differently
        value,bestalpha=self.vf.value(b,parallel=self.parallel)
        if self.set_used:
            bestalpha.used=True
        self.last_value=value
        return bestalpha.action

class OptimalPolicyWithCorr:
    def __init__(self,vf0,vf1,agent,parallel=False,epsilon=0):
        self.vf0=vf0
        self.vf1=vf1
        self.agent=agent
        self.parallel=parallel
        self.set_used=False
        self.epsilon=epsilon
        self.last_value=None
    def getAction(self,belief):
        if random.random()<self.epsilon:
            return random.choice(self.agent.env.actions)
        b=self.agent.perseus_belief(belief) # this line for situations where belief used by perseus is defined differently
        if self.agent.last_obs is None:
            raise RuntimeError('agent has undefined observational state!')
        elif self.agent.last_obs==0:
            value,bestalpha=self.vf0.value(b,parallel=self.parallel)
        elif self.agent.last_obs==1:
            value,bestalpha=self.vf1.value(b,parallel=self.parallel)
        else:
            raise RuntimeError('agent has unrecognized observational state!')
        if self.set_used:
            bestalpha.used=True
        self.last_value=value
        return bestalpha.action

class GreedyPolicy:
    def __init__(self,agent):
        self.agent=agent

    def getAction(self,belief):
        location = np.unravel_index(np.argmax(belief),belief.shape)
        return random.choice(follow_loc(location,self.agent.true_pos))

class SpaceAwareInfotaxis:
    '''
    there are two versions of SAI implemented here. if new=False, then we use a slight modification of the original policy introduced in
    Loisy and Eloy (2022). this modification is described in Heinonen et al. (2023).
    if new=True, then we minimize the expectation of J = log((1-alpha)*exp(H) + alpha*|x-y|_1), so that alpha=0 is infotaxis and alpha=1 is QMDP.
    with_corr=True takes the previous observation and conditional likelihoods into account, but this generally performs poorly for this myopic
    version of SAI (it may work ok if we generalize to look N steps in the future).
    '''
    def __init__(self,agent,epsilon=0,base=None,out_of_bounds_actions=False,with_corr=False,alpha=0.5,new=False,verbose=False,tiebreak='random',tol=0):
        self.agent=agent
        self.epsilon=epsilon
        #self.type=type
        self.alpha=alpha
        self.out_of_bounds_actions=out_of_bounds_actions
        self.with_corr=with_corr
        self.base=base 
        self.new=new
        self.verbose=verbose
        self.tiebreak=tiebreak
        self.tol=tol

    def getAction(self,belief,bad_points=[]):
        if np.random.random()<self.epsilon:
            return np.random.choice(self.ag.env.actions)
        newJ=np.inf
        best_action=[]

        for action in self.agent.env.actions:
            if self.verbose:
                print(action)
            if not self.out_of_bounds_actions:
                if self.agent.env.outOfBounds(self.agent.true_pos+action):
                    continue
                bad=False
                new_pos=self.agent.true_pos+action
                for pt in bad_points:
                    if pt[0]==new_pos[0] and pt[1]==new_pos[1]:
                        bad=True
                        break
                if bad:
                    continue
                    
            newpos=self.agent.belief_env.transition(self.agent.true_pos,action)
            ps=belief[newpos[0],newpos[1]]
            if ps==1:
                best_action=[action]
                break
            x=newpos[0]-np.arange(self.agent.belief_env.dims[0])
            y=newpos[1]-np.arange(self.agent.belief_env.dims[1])
            probs=[]
            for i in range(0,self.agent.belief_env.obs[-1]):
                if self.with_corr:
                    p=self.agent.belief_env.get_likelihood(x[:,None],y[None,:],i,self.agent.last_obs,action)*belief
                else:
                    p=self.agent.belief_env.get_likelihood(x[:,None],y[None,:],i)*belief
                probs.append(np.sum(p))
            probs.append(1-sum(probs)-ps)
            dist=np.abs(x[:,None])+np.abs(y[None,:])
            js=[]
            for i in range(self.agent.belief_env.obs[-1]+1):
                try:
                    if self.with_corr:
                        b=self.agent.computeBelief(i,belief,last_action=action,action=action)
                    else:
                        b=self.agent.computeBelief(i,belief,action=action)
                    if not self.new:                   
                        s=entropy(b.flatten(),base=2)
                        j=np.log2((2**(s-1)+1/2)*(1-self.alpha)*2+self.alpha*2*np.sum(dist*b))
                    else:
                        s=entropy(b.flatten(),base=self.base)
                        if self.base is None:
                            j=np.log(np.exp(s)*(1-self.alpha)+self.alpha*np.sum(dist*b))
                        else:
                            j=np.log(self.base**s*(1-self.alpha)+self.alpha*np.sum(dist*b))/np.log(self.base)
                except:
                    j=9999999
                js.append(j)
            
            tmp=np.sum(np.multiply(js,probs))
            if self.verbose:
                print("expected J: ",tmp)
            if tmp<newJ-self.tol:
                newJ=tmp
                best_action=[action]
            elif np.abs(tmp-newJ)<self.tol:
                best_action.append(action)
        if self.verbose:
            print("best action is: ",best_action)

        if not best_action:
            return np.array([0,0])
        if len(best_action)==4:
            if self.tiebreak=='random':
                return random.choice(best_action)
            if self.tiebreak=='greedy':
                index=np.argmax(belief)
                return random.choice(follow_loc(np.unravel_index(index,belief.shape),self.agent.true_pos))
            raise NotImplementedError('unrecognized tiebreak option for SAI')
        return random.choice(best_action)

class SecondOrderInfotaxisPolicy:
    '''
    a (clunky) implementation of Infotaxis with 2-step lookahead, also allowing to take one-step correlations into account
    '''
    def __init__(self,agent,verbose=False,epsilon=0,out_of_bounds_actions=False,with_corr=False):
        self.agent=agent
        self.epsilon=epsilon
        self.verbose=verbose
        self.out_of_bounds_actions=out_of_bounds_actions
        self.with_corr=with_corr


    def getAction(self,belief):
        if random.random() < self.epsilon:
            return random.choice(ag.env.actions)
        newS=np.inf
        #S=entropy(belief)
        #print("entropy: ",S)
        best_action=[]
        if True:
            for a1 in self.agent.env.actions:
                for a2 in self.agent.env.actions:
                    if self.verbose:
                        print('actions:',a1,a2)
                    probs=[]
                    entropies=[]
                    try:
                        for i in range(0,self.agent.env.numobs):
                            for j in range(0,self.agent.env.numobs):
                                if self.with_corr:
                                    bprime=self.agent.computeBelief(i,belief,a1,action=a1)
                                    posprime=self.agent.belief_env.transition(self.agent.true_pos,a1)
                                    bprimeprime=self.agent.computeBelief(j,bprime,a2,action=a2,last_obs=i)
                                    posprimeprime=self.agent.belief_env.transition(posprime,a2)
                                    x1=posprime[0]-np.arange(self.agent.belief_env.dims[0])
                                    y1=posprime[1]-np.arange(self.agent.belief_env.dims[1])
                                    p1=self.agent.belief_env.get_likelihood(x1[:,None],y1[None,:],i,self.agent.last_obs,a1)
                                    x2=posprimeprime[0]-np.arange(self.agent.belief_env.dims[0])
                                    y2=posprimeprime[1]-np.arange(self.agent.belief_env.dims[1])
                                    p2=self.agent.belief_env.get_likelihood(x2[:,None],y2[None,:],j,i,a2)
                                    probs.append(np.sum(p1*p2*belief))
                                    entropies.append(entropy(bprimeprime.flatten()))
                                else:
                                    bprime=self.agent.computeBelief(i,belief,action=a1)
                                    posprime=self.agent.belief_env.transition(self.agent.true_pos,a1)
                                    bprimeprime=self.agent.computeBelief(j,bprime,action=a2)
                                    posprimeprime=self.agent.belief_env.transition(posprime,a2)
                                    x1=posprime[0]-np.arange(self.agent.belief_env.dims[0])
                                    y1=posprime[1]-np.arange(self.agent.belief_env.dims[1])
                                    p1=self.agent.belief_env.get_likelihood(x1[:,None],y1[None,:],i)
                                    x2=posprimeprime[0]-np.arange(self.agent.belief_env.dims[0])
                                    y2=posprimeprime[1]-np.arange(self.agent.belief_env.dims[1])
                                    p2=self.agent.belief_env.get_likelihood(x2[:,None],y2[None,:],j)
                                    probs.append(np.sum(p1*p2*belief))
                                    entropies.append(entropy(bprimeprime.flatten()))

                        tmp=np.sum(np.multiply(probs,entropies))
                    except:
                        tmp=999999
                    if self.verbose:
                        print("expected entropy: ",tmp)
                        print("terms:",[e*p for e,p in zip(entropies,probs)])
                    if tmp<newS:
                        best_action=[a1]
                        newS=tmp




        else:
            for action in self.agent.env.actions:
                if self.verbose:
                    print('action',action)
                if not self.out_of_bounds_actions:
                    if self.agent.env.outOfBounds(self.agent.true_pos+action):
                        if self.verbose:
                            print('out of bounds')
                        continue
                newpos=self.agent.belief_env.transition(self.agent.true_pos,action)
                ps=belief[newpos[0],newpos[1]]
                x=newpos[0]-np.arange(self.agent.belief_env.dims[0])
                y=newpos[1]-np.arange(self.agent.belief_env.dims[1])
                probs=[]
                for i in range(0,self.agent.belief_env.obs[-1]):
                    if self.with_corr:
                        p=self.agent.belief_env.get_likelihood(x[:,None],y[None,:],i,self.agent.last_obs,action)*belief
                    else:
                        p=self.agent.belief_env.get_likelihood(x[:,None],y[None,:],i)*belief
                    probs.append(np.sum(p))
                probs.append(1-sum(probs)-ps)
                entropies=[]
                for i in range(self.agent.belief_env.obs[-1]+1):
                    if self.with_corr:
                        b=self.agent.computeBelief(i,belief,action,action=action)
                    else:
                        b=self.agent.computeBelief(i,belief,action=action)
                    s=entropy(b.flatten())
                    entropies.append(s)
                tmp=np.sum(np.multiply(entropies,probs))
                #s0=entropy(self.agent.computeBelief(False,belief,action).flatten())
                #print("entropy associated with no hit:",s0)
                #s1=entropy(self.agent.computeBelief(True,belief,action).flatten())
                #print("entropy associated with hit:",s1)
                #tmp=l_miss*s0+l_hit*s1
                if self.verbose:
                    print("expected entropy: ",tmp)
                    print("terms:",[e*p for e,p in zip(entropies,probs)])
                if tmp<newS:
                    newS=tmp
                    best_action=[action]
                elif tmp==newS:
                    best_action.append(action)
        #print("best action is: ",best_action)

        if best_action is None:
            raise RuntimeError('Failed to find optimal action')
        return random.choice(best_action)


class InfotacticPolicy:
    '''
    infotaxis, introduced by Vergassola, Villermaux, and Shraiman (2007)
    exponents is an experimental feature and should generally not be used
    '''
    def __init__(self,agent,verbose=False,epsilon=0,out_of_bounds_actions=False,with_corr=False,tiebreak='random',exponents=None):
        self.agent=agent
        self.epsilon=epsilon
        self.verbose=verbose
        self.out_of_bounds_actions=out_of_bounds_actions
        self.with_corr=with_corr
        self.tiebreak=tiebreak
        self.exponents=exponents
       
    def getAction(self,belief):
        if random.random() < self.epsilon:
            return random.choice(ag.env.actions)
        newS=np.inf
        #S=entropy(belief)
        #print("entropy: ",S)
        best_action=[]
        for action_index,action in enumerate(self.agent.env.actions):
            if self.verbose:
                print('action',action)
            if not self.out_of_bounds_actions:
                if self.agent.env.outOfBounds(self.agent.true_pos+action):
                    if self.verbose:
                        print('out of bounds')
                    continue
            newpos=self.agent.belief_env.transition(self.agent.true_pos,action)
            ps=belief[newpos[0],newpos[1]]
            if ps==1:
                best_action=[action]
                break
            #print("probability of finding source: ",p)
            x=newpos[0]-np.arange(self.agent.belief_env.dims[0])
            y=newpos[1]-np.arange(self.agent.belief_env.dims[1])
            probs=[]
            for i in range(0,self.agent.belief_env.obs[-1]):
                if self.with_corr:
                    p=self.agent.belief_env.get_likelihood(x[:,None],y[None,:],i,self.agent.last_obs,action)*belief
                else:
                    p=self.agent.belief_env.get_likelihood(x[:,None],y[None,:],i)*belief
                probs.append(np.sum(p))
            probs.append(1-sum(probs)-ps)
            entropies=[]
            for i in range(self.agent.belief_env.obs[-1]+1):
                if self.exponents is None:
                    exponent=None
                else:
                    exponent=self.exponents[action_index]
                try:
                    if self.with_corr:
                        b=self.agent.computeBelief(i,belief,action,action=action,exponent=exponent)
                    else:
                        b=self.agent.computeBelief(i,belief,action=action,exponent=exponent)
                    s=entropy(b.flatten())
                except:
                    #raise RuntimeError()
                    s=9999999
                entropies.append(s)
            tmp=np.sum(np.multiply(entropies,probs))
            #s0=entropy(self.agent.computeBelief(False,belief,action).flatten())
            #print("entropy associated with no hit:",s0)
            #s1=entropy(self.agent.computeBelief(True,belief,action).flatten())
            #print("entropy associated with hit:",s1)
            #tmp=l_miss*s0+l_hit*s1
            if self.verbose:
                print("expected entropy: ",tmp)
            if tmp<newS:
                newS=tmp
                best_action=[action]
            elif tmp==newS:
                best_action.append(action)
        #print("best action is: ",best_action)

        if best_action is None:
            raise RuntimeError('Failed to find optimal action')
        if len(best_action)==4:
            if self.tiebreak=='random':
                return random.choice(best_action)
            if self.tiebreak=='greedy':
                index=np.argmax(belief)
                return random.choice(follow_loc(np.unravel_index(index,belief.shape),ag.true_pos))
            raise NotImplementedError('unrecognized tiebreak option for infotaxis')

        return random.choice(best_action)


def follow_loc(location,true_pos):
    if location[1]==true_pos[1]:
        if location[0]>true_pos[0]:
            return [np.array([1,0])]
        return [np.array([-1,0])]
    if location[0]==true_pos[0]:
        if location[1]>true_pos[1]:
            return [np.array([0,1])]
        return [np.array([0,-1])]
    x=[]
    if location[0]>true_pos[0]:
        x.append(np.array([1,0]))
    else:
        x.append(np.array([-1,0]))
    if location[1]>true_pos[1]:
        x.append(np.array([0,1]))
    else:
        x.append(np.array([0,-1]))
    return x
