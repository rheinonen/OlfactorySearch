import numpy as np
import scipy.special as sc
from datetime import datetime
import random
import math
#scipy.special.cython_special as spec
from scipy.interpolate import interp1d

class OlfactorySearch2D:
    '''
    base environment class for olfactory search. the likelihood will be input as an array.
    dims is a tuple of integers defining the size of the gridworld.
    dummy is True if observations are drawn stochastically from the model, and False if they
    are drawn from concentration data. in the latter case, an obseration threshold should be set.
    corr is True if the model likelihood should depend on previous observations.
    obs_radius and min_ell are experimental features and should generally be set to defaults.
    '''
    def __init__(self,dims,corr=True,obs_radius=0,min_ell=0,dummy=False,**kwargs):
        self.dummy=dummy
        self.obs_radius=obs_radius # used for low-info limit
        self.min_ell=min_ell # used for low-info limit
        if 'data' in kwargs:
            self.conc_data=kwargs['data']
        else:
            self.conc_data=None
        if 'threshold' in kwargs:
            self.threshold=kwargs['threshold']
        else:
            self.threshold=None
        self.t=0
        self.data_r0=None #position of source in concentration data

        self.dims=dims # number of (x,y) gridpoints. Ny should be odd
        if 'x0' in kwargs:
            self.x0=kwargs["x0"] #integer downwind position of source relative to lefthand boundary
        else:
            self.x0=None
        if 'y0' in kwargs:
            self.y0=kwargs["y0"] #integer downwind position of source relative to righthand boundary
        else:
            self.y0=None
        self.actions=[np.array([1,0]),np.array([-1,0]),np.array([0,1]),np.array([0,-1])]
        self.numobs=2 #only implementing a single threshold for now
        self.obs=[i for i in range(-1,2)]
        if self.x0 is not None and self.y0 is not None:
            self.pos=np.array([self.x0,self.y0])
        if 'gamma' in kwargs:
            self.gamma=kwargs['gamma']
        else:
            self.gamma=1
        self.likelihood=None
        self.agent=None
        self.unconditional_likelihood=None
        self.corr=corr
        self.tstep=1
        if "tstep" in kwargs:
            self.tstep = kwargs['tstep']
        self.obs_per_action=1
        self.actions_per_obs=1
        self.numactions=len(self.actions)
        
        dist=np.zeros((2*self.dims[0]-1,2*self.dims[1]-1))
        dist_l2=np.zeros((2*self.dims[0]-1,2*self.dims[1]-1))
        for i in range(-(self.dims[0]-1),self.dims[0]):
            for j in range(-(self.dims[1]-1),self.dims[1]):
                dist[i,j]=np.abs(i)+np.abs(j)
                dist_l2[i,j]=np.sqrt(i**2+j**2)
        self.dist=dist
        self.dist_l2=dist_l2
        self.dist_l2=np.roll(dist_l2,(-self.dims[0],-self.dims[1]),axis=(0,1))
        self.reward1=np.zeros((2*self.dims[0]-1,2*self.dims[1]-1))
        self.reward2=np.zeros((2*self.dims[0]-1,2*self.dims[1]-1))
        self.reward3=np.zeros((2*self.dims[0]-1,2*self.dims[1]-1))
        self.reward4=np.zeros((2*self.dims[0]-1,2*self.dims[1]-1))

        self.reward1[-1,0]=1
        self.reward2[1,0]=1
        self.reward3[0,-1]=1
        self.reward4[0,1]=1

        self.rewards=[self.reward1,self.reward2,self.reward3,self.reward4]
    
    def set_likelihood(self,l_un,l0=None,l1=None,sim_r0=[93,16],l_static_0=None,l_static_1=None):
        '''
        l_un is the unconditional (single-time) likelhihood of a detection.
        l0 and l1 are lists of likelihood (of detection) matrices for each action, given an observational state,
        given as function of FINAL position (AFTER the action). not necessary unless you are using a dummy 
        1-step Markov environment or you want the agent to be aware of 1-step correlations.
        sim_r0 is the location of the source within the inputted array.
        l_static_0 and l_static_1 are typically not used but refer to the likelihoods conditioned on not moving.
        '''
        xs=np.arange(self.dims[0]-1-sim_r0[0],2*self.dims[0]-1-sim_r0[0])
        ys=np.arange(self.dims[1]-1-sim_r0[1],2*self.dims[1]-1-sim_r0[1])
        if self.corr:
            out0=[]
            out1=[]
            for a in range(self.numactions):
                l0_out=np.zeros((2*self.dims[0]-1,2*self.dims[1]-1))
                l1_out=np.zeros((2*self.dims[0]-1,2*self.dims[1]-1))
                l0_out[xs[:,None],ys[None,:]]=l0[a]
                l1_out[xs[:,None],ys[None,:]]=l1[a]
                out0.append(l0_out.copy())
                out1.append(l1_out.copy())
            if l_static_0 is not None:
                l_static_0_out=np.zeros((2*self.dims[0]-1,2*self.dims[1]-1))
                l_static_1_out=np.zeros((2*self.dims[0]-1,2*self.dims[1]-1))
                l_static_0_out[xs[:,None],ys[None,:]]=l_static_0
                l_static_1_out[xs[:,None],ys[None,:]]=l_static_1
                out0.append(l_static_0_out.copy())
                out1.append(l_static_1_out.copy())
            self.likelihood=[out0,out1]
        else:
            self.likelihood=None    
        out_un=np.zeros((2*self.dims[0]-1,2*self.dims[1]-1))
        out_un[xs[:,None],ys[None,:]]=l_un
        if self.obs_per_action>1:
            ll=[]
          #  for i in range(self.obs_per_action+1):
           #     new_l=(out_un**i)*(1-out_un)**(self.obs_per_action-i)*(sc.comb(self.obs_per_action,i)
            #    ll.append(new_l)
        else:
            self.unconditional_likelihood=out_un

    def stepInTime(self):
        self.t+=self.tstep
    
    def reset(self):
        self.t=0
    
    def set_pos(self,x0,y0,):
        self.x0=x0
        self.y0=y0
        self.pos=np.array([x0,y0])

    def getObs(self,pos,ag=None):
        if self.dummy:
            if self.corr:
                l=self.get_likelihood(pos[0]-self.x0,pos[1]-self.y0,1,ag.last_obs,ag.last_action)
            else:
                l=self.get_likelihood(pos[0]-self.x0,pos[1]-self.y0,1)
            if l<self.min_ell:
                return 0
         #   print("prob of obs is",l)
            return int(random.random()<l)

        data_pos=self.data_r0+pos-np.array([self.x0,self.y0])
        if data_pos[0]<0 or data_pos[1]<0 or data_pos[0]>=self.dims[0] or data_pos[1]>=self.dims[1]:
            return 0
        if int(self.t)-self.t==0:
            return int(self.conc_data[int(self.t),data_pos[0],data_pos[1]]>=self.threshold)
        conc=self.conc_data[int(np.floor(self.t)),data_pos[0],data_pos[1]]*(1-self.t+np.floor(self.t))+self.conc_data[int(np.ceil(self.t)),data_pos[0],data_pos[1]]*(self.t-np.floor(self.t))
        return int(conc>=self.threshold)


    def set_data(self,data,data_r0=[93,16]):
        '''
        if dummy=False, this must be called to set the concentration data
        '''
        self.conc_data=data
        self.data_r0=data_r0

    def array_to_int(self,action):
        if np.array_equal(action,[1,0]):
            return 0
        if np.array_equal(action,[-1,0]):
            return 1
        if np.array_equal(action,[0,1]):
            return 2
        if np.array_equal(action,[0,-1]):
            return 3
        if np.array_equal(action,[0,0]):
            return 4

    def get_likelihood(self,x,y,obs,obs_state=None,action=None,low_info=False):
        if low_info:
            l=(self.dist_l2<=self.obs_radius).astype('int')
            l=l[x+self.dims[0]-1,y+self.dims[1]-1]
        elif action is None:
            l=self.unconditional_likelihood[x+self.dims[0]-1,y+self.dims[1]-1]
        else:
            l=self.likelihood[obs_state][self.array_to_int(action)][x+self.dims[0]-1,y+self.dims[1]-1]
        if obs==0:
            out=1-l
        else:
            out=l
        if isinstance(out,np.ndarray):
            origin_x=np.argwhere(np.squeeze(x)==0)
            origin_y=np.argwhere(np.squeeze(y)==0)
            out[origin_x,origin_y]=0
        return out

    def transition(self,pos,action):
        tmp=pos+action
        if not self.outOfBounds(tmp):
            return tmp # agent is unmoved if attempts to leave simulation bounds
        return pos

    def outOfBounds(self,pos):
        if (pos[0]<0 or pos[0]>self.dims[0]-1 or pos[1] < 0 or pos[1] > self.dims[1]-1):
            return True
        return False

    def getReward(self,true_pos,action):
        r=true_pos-self.pos
        # if (action==np.array([0,0])).all():
        #     return self.rewards[0][r[0],r[1]]
        if (action==np.array([1,0])).all():
            return self.rewards[0][r[0],r[1]]
        elif (action==np.array([-1,0])).all():
            return self.rewards[1][r[0],r[1]]
        elif (action==np.array([0,1])).all():
            return self.rewards[2][r[0],r[1]]
        elif (action==np.array([0,-1])).all():
            return self.rewards[3][r[0],r[1]]

class TigerGrid:
    def __init__(self,gamma=0.95):
        self.seed=datetime.now()
        random.seed(self.seed)
        self.t=0
        self.numactions=3
        self.actions=[0,1,2] # forward, rotate right, rotate left
        # state is encoded by an integer 0-32 given by square index*4+direction (0 up, 1 right, 2 down, 3 left)
        self.gamma=gamma
        self.dims=(33,)
        self.numobs=5 #0 is nothing. 1 is wall. 2 is tiger. 3 is tiger+wall. 4 is reward

        reward0=np.zeros((33,))
        reward1=np.zeros((33,))
        reward2=np.zeros((33,))
        reward0[5]=100
        reward0[7]=-100
        reward0[9]=-100
        reward0[11]=100
        reward0[16]=-100
        reward0[28]=-100
        reward0[0]=-100
        reward0[3]=-100
        reward0[12]=-100
        reward1[0]=-100
        reward1[1]=-100
        reward1[2]=-100
        reward1[3]=-100
        reward2[0]=-100
        reward2[1]=-100
        reward2[2]=-100
        reward2[3]=-100
        reward1[12]=-100
        reward1[13]=-100
        reward1[14]=-100
        reward1[15]=-100
        reward2[12]=-100
        reward2[13]=-100
        reward2[14]=-100
        reward2[15]=-100
        self.rewards=[reward0,reward1,reward2]
        self.pobs=np.zeros((5,33))
        for i in range(33):
            j=self.getObs(i)
            self.pobs[j,i]=1
        self.ptrans=np.zeros((3,33,33)) # action, s, s'
        for a in range(3):
            for s in range(32):
                sp=self.transition(s,a)
                self.ptrans[a,s,sp]=1
        self.ptrans[:,-1,:-1]=1/32


    def stepInTime(self):
        self.t+=1

    def transition(self,state,action):
        if state==32:
            return random.randrange(32)
        direction=state%4
        pos=state//4
        if action==0:
            if self.facingWall(state):
                return state
            elif direction==0:
                pos=pos-4
            elif direction==1:
                if pos==1:
                    return 32
                pos=pos+1
            elif direction==2:
                pos=pos+4
            elif direction==3:
                if pos==2:
                    return 32
                pos=pos-1

        else:
            if action==1:
                direction=(direction+1)%4
            if action==2:
                direction=(direction-1)%4
        return pos*4+direction

    def getObs(self,state):
        if state==32:
            return 4
        direction=state%4
        pos=state//4
        wall=self.facingWall(state)
        if (pos==0 or pos==3) and wall:
            return 3
        if pos==0 or pos==3:
            return 2
        if wall:
            return 1
        return 0

    def getReward(self,action):
        return self.rewards[action]

    def get_g(self,alpha,action):
        ptrans=np.squeeze(self.ptrans[action,:,:])
        g=np.einsum('ij,kj,j->ki',ptrans,self.pobs,alpha)
        out=[]
        for i in range(5):
            out.append(np.squeeze(g[i,:]))
        return out

    def facingWall(self,state):
        direction=state%4
        pos=state//4
        if pos<4 and direction==0:
            return True
        if pos>=4 and direction==2:
            return True
        if (pos==0 or pos==4) and direction==3:
            return True
        if (pos==3 or pos==7) and direction==1:
            return True
        if pos==5 and direction==1:
            return True
        if pos==6 and direction==3:
            return True
        return False
#
class BernoulliBandits:
    def __init__(self,probs,Np=51,gamma=0.9):
        self.seed=datetime.now()
        random.seed(self.seed)
        self.agent=None
        self.t=0
        self.numobs=2
        self.gamma=gamma
        #self.rewards=rewards
        self.p_grid=np.linspace(0,1,Np)
        self.probs=probs # list specifying probablities of success for each arm
        self.numactions=len(probs)
        self.dims=(Np,)*self.numactions
        self.actions=[a for a in range(0,self.numactions) ]
        self.rewards=[]
        for a in range(0,self.numactions):
            x=self.p_grid
            x=np.broadcast_to(x,self.dims)
            perm=[i for i in range(0,self.numactions-1)]
            perm.insert(a,self.numactions-1)
            perm=tuple(perm)
            x=np.transpose(x,perm)
            self.rewards.append(x)

    def stepInTime(self):
        self.t+=1

    def set_agent(self,agent):
        self.agent=agent

    def getReward(self):
        #EXPECTED rewards based on current belief
        #not clear this function will be used
        return self.agent.alphas/(self.agent.alphas+self.agent.betas)

    def getObs(self,action):
        return random.random()<self.probs[action]

    def get_g(self,alpha,action):
        shape=(1,)*action+self.p_grid.shape+(1,)*(self.numactions-action-1)
        x=np.reshape(self.p_grid,shape)
        return [x*alpha,(1-x)*alpha]

class Tag:
    def __init__(self,gamma=0.95):
        self.seed=datetime.now()
        random.seed(self.seed)
        self.t=0
        self.numactions=5
        self.actions=['n','s','e','w','tag']

        self.gamma=gamma
        self.dims=(29,30)
        self.numobs=2
        reward_tag=-10*np.ones(self.dims)
        for i in range(29):
            reward_tag[i,i]=10
        self.rewards=[-1*np.ones(self.dims),-1*np.ones(self.dims),-1*np.ones(self.dims),-1*np.ones(self.dims),reward_tag]
        p_trans=np.zeros((5,29,30,29,30))

        for a in range(5):
            for s11 in range(29):
                for s12 in range(30):
                    for s21 in range(29):
                        for s22 in range(30):
                            p_trans[a,s11,s12,s21,s22]=self.p_trans(self.actions[a],(s11,s12),(s21,s22))

        for a in range(5):
            for s11 in range(29):
                for s12 in range(30):
                    #print(a,s11,s12)
                    #print(np.sum(p_trans[a,s11,s12,:,:]))
                    assert(np.sum(p_trans[a,s11,s12,:,:])==1)

        self.pt=p_trans
        self.p_obs=[np.zeros((29,30)),np.zeros((29,30))]
        for i in range(29):
            for j in range(30):
                if i==j:
                    self.p_obs[1][i,j]=1
        self.p_obs[0]=1-self.p_obs[1]
        self.p_obs[0][:,29]=0

        self.pos=random.randrange(29)

    def p_trans(self,action,state1,state2):
        if state1[1]==29:
            if state2[1]==29 and state1[0]==state2[0]:
                return 1
            return 0
        if action=='tag':
            if state1[0]==state2[0]:
                if state1[0]==state1[1]:
                    if state2[1]==29:
                        return 1
                else:
                    return self.p_run(state1,state2[1])
            return 0
        if state2[0]==self.chase(state1[0],action):
            return self.p_run(state1,state2[1])
        return 0

    def transition_function(self,belief,action):
        if action=='n':
            a=0
        if action=='s':
            a=1
        if action=='e':
            a=2
        if action=='w':
            a=3
        if action=='tag':
            a=4
        t0=np.einsum('ij,ijlm,lm->lm',belief,np.squeeze(self.pt[a,:,:,:,:]),self.p_obs[0])
        t1=np.einsum('ij,ijlm,lm->lm',belief,np.squeeze(self.pt[a,:,:,:,:]),self.p_obs[1])
        if np.sum(t0)==0:
            t0_out=np.zeros_like(t0)
        else:
            t0_out=t0/np.sum(t0)
        if np.sum(t1)==0:
            t1_out=np.zeros_like(t1)
        else:
            t1_out=t1/np.sum(t1)

        return [np.sum(t0),np.sum(t1)],[t0_out,t1_out]

    def p_run(self,state,s2):
        if s2==state[1]:
            return 0.2
        actions=[]
        if self.x_coord(state[0])<=self.x_coord(state[1]):
            actions.append('e')
        if self.x_coord(state[0])>=self.x_coord(state[1]):
            actions.append('w')
        if self.y_coord(state[0])<=self.y_coord(state[1]):
            actions.append('n')
        if self.y_coord(state[0])>=self.y_coord(state[1]):
            actions.append('s')

        good_actions=[]
        for a in actions:
            if not self.out_of_bounds(state[1],a):
                good_actions.append(a)
        if not good_actions:
            actions=['e','w','n','s']
            for a in actions:
                if not self.out_of_bounds(state[1],a):
                    good_actions.append(a)
        num=len(good_actions)
        for a in good_actions:
            if s2==self.chase(state[1],a):
                return 1./num*0.8
        return 0


    def transition(self,state,action):
        #print("state is",state)
        chaser_state=state[0]
        opponent_state=state[1]
        if chaser_state==opponent_state and action=='tag':
            return (chaser_state,29) # chaser wins
        if action=='tag':
            return (chaser_state,self.run_away(state)) #chaser fucced up
        return (self.chase(state[0],action),self.run_away(state))

    def x_coord(self,state):
        if state<20:
            return state%10
        return (state-20)%3+5
    def y_coord(self,state):
        if state<20:
            return state//10
        return state//10+(state-20)//3

    def run_away(self,state):
        if random.random()<0.2:
            return state[1]
        actions=[]
        if self.x_coord(state[0])<=self.x_coord(state[1]):
            actions.append('e')
        if self.x_coord(state[0])>=self.x_coord(state[1]):
            actions.append('w')
        if self.y_coord(state[0])<=self.y_coord(state[1]):
            actions.append('n')
        if self.y_coord(state[0])>=self.y_coord(state[1]):
            actions.append('s')

        good_actions=[]
        for a in actions:
            if not self.out_of_bounds(state[1],a):
                good_actions.append(a)
        if not good_actions:
            actions=['e','w','n','s']
            for a in actions:
                if not self.out_of_bounds(state[1],a):
                    good_actions.append(a)

        action=random.choice(good_actions)
        return self.chase(state[1],action)

    def out_of_bounds(self,state,action):
        if state==0 and action=='w':
            return True
        if state==9 and action=='e':
            return state
        if state<10 and action=='s':
            return True
        if state<20 and state>=10:
            if state<15 and state>17 and action=='n':
                return True
            if state==10 and action=='w':
                return True
            if state==19 and action=='e':
                return True
        if state==20 and action=='w':
            return True
        if state==22 and action=='e':
            return True
        if state==23 and action=='w':
            return True
        if state==25 and action=='e':
            return True
        if state>=26 and action=='n':
            return True
        if state==26 and action=='w':
            return True
        if state==28 and action=='e':
            return True
        return False

    def chase(self,state,action):
        if self.out_of_bounds(state,action):
            return state
        if action=='e':
            return state+1
        if action=='w':
            return state-1
        if state<10 and action=='n':
            if action=='n':
                return state+10
        if state<20:
            if action=='n':
                return state+5
            if action=='s':
                return state-10
        if state<23:
            if action=='s':
                return state-5
            if action=='n':
                return state+3
        if state<26:
            if action=='s':
                return state-3
            if action=='n':
                return state+3
        if action=='s':
            return state-3


    def stepInTime(self):
        self.t+=1


    def getObs(self,state):
        if state[0]==state[1]:
            return True
        return False

    def get_g(self,alpha,action):
        if action=='n':
            a=0
        if action=='s':
            a=1
        if action=='e':
            a=2
        if action=='w':
            a=3
        if action=='tag':
            a=4
        ptrans=np.squeeze(self.pt[a,:,:,:,:])
        g1=np.einsum('ijkl,kl,kl->ij',ptrans,self.p_obs[0],alpha)
        g2=np.einsum('ijkl,kl,kl->ij',ptrans,self.p_obs[1],alpha)
        return [g1,g2]

    def getReward(self,state,action):
        if action=='tag':
            if state[0]==state[1]:
                return 10
            return -10
        return -1

    def reset(self,r=None):
        if r is None:
            self.pos=random.randrange(29)
        else:
            self.pos=r
        self.t=0

def unravel_belief(b):
    out=np.zeros_like(b)
    dims0=(out.shape[0]+1)//2
    dims1=(out.shape[1]+1)//2
    for i in range(-(dims0-1),dims0):
        for j in range(-(dims1-1),dims1):
            new_i=i+dims0-1
            new_j=j+dims1-1
            out[new_i,new_j]=b[i,j]
    return out
