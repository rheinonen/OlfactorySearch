import numpy as np
import pickle
import perseus_redux
import os

alphas=[]
alphas0=[]
alphas1=[]
x_shape=int(os.environ.get('SHAPE_X'))
y_shape=int(os.environ.get('SHAPE_Y'))
dims=(2*x_shape-1,2*y_shape-1)

class alpha_vec:
    def __init__(self,alph,a,g=None):
        self.data=alph
        if g is not None:
            self.gs=g
        else:
            self.gs=None
        self.action=a
        self.used=0

    def __eq__(self, other):
         return np.array_equal(self.data,other.data)

policy_dir=str(os.environ.get('POLICY_DIR'))
in_name=str(os.environ.get('RAW_POLICY'))
if 'CORR_POL' in os.environ:
    with_corr=bool(int(os.environ.get('CORR_POL')))
else:
    with_corr=False

if with_corr:
    with open(policy_dir+'/'+in_name,'r') as f:
        lines=f.readlines()
        lines.pop(0)
        lines.pop(0)
        lines.pop(0)
        lines.pop(-1)
        for line in lines:
            a=int(line[16])
            if a==0:
                action=np.array([1,0])
            if a==1:
                action=np.array([-1,0])
            if a==2:
                action=np.array([0,1])
            if a==3:
                action=np.array([0,-1])
            line=line[32:-10]
            line = [float(i) for i in line.split()]
            
            arr=np.array(line)
            arr0=arr[:dims[0]*dims[1]].reshape(dims)
            arr1=arr[dims[0]*dims[1]:].reshape(dims)
            alphas0.append(perseus_redux.alpha_vec(arr0,action))
            alphas1.append(perseus_redux.alpha_vec(arr1,action))
    alphas={
    'alphas_0':alphas0,
    'alphas_1':alphas1
    }
    out_name=str(os.environ.get('POLICY_FILE'))

    with open(policy_dir+'/'+out_name,'wb') as f:
        pickle.dump(alphas,f)
else:
    with open(policy_dir+'/'+in_name,'r') as f:
        lines=f.readlines()
        lines.pop(0)
        lines.pop(0)
        lines.pop(0)
        lines.pop(-1)
        for line in lines:
            a=int(line[16])
            if a==0:
                action=np.array([1,0])
            if a==1:
                action=np.array([-1,0])
            if a==2:
                action=np.array([0,1])
            if a==3:
                action=np.array([0,-1])
            line=line[32:-10]
            line = [float(i) for i in line.split()]
            arr=np.array(line).reshape(dims)
            alphas.append(perseus_redux.alpha_vec(arr,action))

    out_name=str(os.environ.get('POLICY_FILE'))
    with open(policy_dir+'/'+out_name,'wb') as f:
        pickle.dump(alphas,f)


