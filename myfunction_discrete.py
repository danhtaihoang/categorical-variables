import numpy as np
from scipy import linalg

#===============================================================================
# s_av: mean value for each bin ; s1_av: possible values of s_av
#===============================================================================
def data_binning(s,nbin):
    l = len(s)
    nbin1 = nbin+1
    s_min = np.min(s)
    s_max = np.max(s)

    s_bin=np.linspace(s_min,s_max,nbin1)
    #print('Sbin:',s_bin)

    ibin = np.digitize(s, s_bin, right=True)
    # set min value to bin 1, not 0:
    for i in range(len(ibin)):
        if ibin[i] < 1:
            ibin[i] = 1
    #print('ibin:',ibin)

    # mean of each bin
    #s1_av = [s[ibin == i].mean() for i in range(1, len(s_bin))]

    #s_sum=[s[ibin==i].sum() for i in range(1,nbin1)]
    #print(s_sum)

    #ibin_count=[list(ibin).count(i) for i in range(1,nbin1)]
    #print(ibin_count)

    s1_av = np.zeros(nbin)
    for i in range(nbin):
        s1_av[i]=(s_bin[i+1]+s_bin[i])/2 # mean value
        #if ibin_count[i] > 0:
        #    s1_av[i] = s_sum[i]/ibin_count[i]
        #else:
        #    s1_av[i] = (s_bin[i+1] + s_bin[i])/2 # median value
    #print(s1_av)

    # set observed value to mean value of each bin
    s_av = [s1_av[ibin[i]-1] for i in range(0,l)]

    return s_av,s1_av

##==============================================================================
#  2018.07.26: model expectation value of s:
#  s[:] processed-sequence, sk[:] possible values of s, w[:] coupling, h0 external local field
##==============================================================================
def model_expectation(s,sk,w,h0):
    l = s.shape[0]
    nbin = len(sk)
    p = np.empty((l,nbin)) ; p1 = np.empty((l,nbin)) ; s_model = np.zeros(l)
    
    for k in range(nbin):
        p[0:l,k] = np.exp(sk[k]*(h0+np.matmul(s[0:l,:],w[:])))
        p1[0:l,k] = sk[k]*p[0:l,k]
    s_model[0:l] = np.sum(p1[0:l,:],axis=1)/np.sum(p[0:l,:],axis=1)
    return s_model

##==============================================================================
## 2018.07.26: from s,sbin --> w, h0    
##==============================================================================
def fit_interaction(s0,sbin,nloop):
    s = s0[:-1]
    l,n = s.shape
    
    m = s.mean(axis=0)
    ds = s - m    
    c = np.cov(ds,rowvar=False,bias=True)
    c_inv = linalg.inv(c)
    
    dst = ds.T
    
    #--------------------------------
    # initial guess
    w_ini = np.random.normal(0.0,1./np.sqrt(n),size=(n,n))
    h_all = np.matmul(s,w_ini.T)
    
    W = np.empty((n,n)) ; H0 = np.empty(n)
    for i0 in range(n):
        s1 = s0[1:,i0]
        cost = np.full(nloop,100.) 
        h = h_all[:,i0]
        
        for iloop in range(1,nloop):
            h_av = np.mean(h)
            hs_av = np.matmul(dst,h-h_av)/l
            w = np.matmul(hs_av,c_inv)
            h0=h_av-np.sum(w*m)
            h = np.matmul(s[:,:],w[:]) + h0
            
            s_model = model_expectation(s,sbin[:,i0],w,h0)
            #s_model = np.tanh(h)
            #s_model = 0.5*np.tanh(0.5*h)
            cost[iloop]=np.mean((s1[:]-s_model[:])**2)
            
            #MSE = np.mean((w[:]-W0[i0,:])**2)
            #slope = np.sum(W0[i0,:]*w[:])/np.sum(W0[i0,:]**2)
            #print(i0,iloop,cost[iloop]) #,MSE,slope)
            
            if cost[iloop] >= cost[iloop-1]: break
            
            #h = h*s1/s_model            
            h *= np.divide(s1,s_model, out=np.zeros_like(s1), where=s_model!=0)
            
            W[i0,:] = w[:]
            H0[i0] = h0

    return W,H0

##==============================================================================
## 2017.07.26: generate sequence
# sk: possible values of s (bin values)     
##==============================================================================
def generate_data(w,h0,sk,l):
    n = w.shape[0]
    nbin = sk.shape[0]
    s = np.full((l,n),100.)
    
    # ini config (at t = 0)
    for i in range(n):
        s[0,i]=sk[np.random.randint(0,nbin),i]

    p1=np.empty(nbin)
    for t in range(l-1):
        for i in range(n):
            for k in range(nbin):
                p1[k]=np.exp(sk[k,i]*(h0[i]+np.sum(w[i,:]*s[t,:])))
            p2=np.sum(p1)

            while s[t+1,i] == 100.:
                k0=np.random.randint(0,nbin)
                p=p1[k0]/p2
                if p>np.random.rand():
                    s[t+1,i]=sk[k0,i]
    return s

##==============================================================================
## cij = <delta_si(t+1) delta_sj(t)>
##==============================================================================
def cross_cov(a,b):
   da = a - np.mean(a, axis=0)
   db = b - np.mean(b, axis=0)
   return np.matmul(da.T,db)/a.shape[0]

def convert_binary(s):
    l,n = np.shape(s)
    s[:,:] = 1.
    for t in range(l):
        for i in range(n):
            if s[t,i] < 0:
                s[t,i] = -1.
    return s

def rescale(s):
    l,n = np.shape(s)
    for i in range(n):
        s[:,i] = (s[:,i] - np.mean(s[:,i]))
        s[:,i] = s[:,i]/np.max(np.abs(s[:,i]))
    return s