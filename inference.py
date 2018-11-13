##========================================================================================
import numpy as np
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder

#=========================================================================================
def itab(n,m):    
    i1 = np.zeros(n)
    i2 = np.zeros(n)
    for i in range(n):
        i1[i] = i*m
        i2[i] = (i+1)*m

    return i1.astype(int),i2.astype(int)
#=========================================================================================
# generate coupling matrix w0: wji from j to i
def generate_interactions(n,m,g,sp):
    nm = n*m
    w = np.random.normal(0.0,g/np.sqrt(nm),size=(nm,nm))
    i1tab,i2tab = itab(n,m)

    # sparse
    for i in range(n):
        for j in range(n):
            if (j != i) and (np.random.rand() < sp): 
                w[i1tab[i]:i2tab[i],i1tab[j]:i2tab[j]] = 0.
    
    # sum_j wji to each position i = 0                
    for i in range(n):        
        i1,i2 = i1tab[i],i2tab[i]              
        w[:,i1:i2] -= w[:,i1:i2].mean(axis=1)[:,np.newaxis]            

    # no self-interactions
    for i in range(n):
        i1,i2 = i1tab[i],i2tab[i]
        w[i1:i2,i1:i2] = 0.   # no self-interactions

    # symmetry
    for i in range(nm):
        for j in range(nm):
            if j > i: w[i,j] = w[j,i]       
        
    return w
#=========================================================================================
def generate_external_local_field(n,m,g):  
    nm = n*m
    h0 = np.random.normal(0.0,g/np.sqrt(nm),size=nm)

    i1tab,i2tab = itab(n,m) 
    for i in range(n):
        i1,i2 = i1tab[i],i2tab[i]
        h0[i1:i2] -= h0[i1:i2].mean(axis=0)

    return h0
#=========================================================================================
# 2018.10.27: generate time series by MCMC
def generate_sequences(w,h0,n,m,l): 
    i1tab,i2tab = itab(n,m)
    
    # initial s (categorical variables)
    s_ini = np.random.randint(0,m,size=(l,n)) # integer values

    # onehot encoder 
    enc = OneHotEncoder(n_values=m)
    s = enc.fit_transform(s_ini).toarray()

    nrepeat = 5*n
    for irepeat in range(nrepeat):
        for i in range(n):
            i1,i2 = i1tab[i],i2tab[i]

            h = h0[np.newaxis,i1:i2] + s.dot(w[:,i1:i2])  # h[t,i1:i2]

            k0 = np.argmax(s[:,i1:i2],axis=1)
            for t in range(l):
                k = np.random.randint(0,m)                
                while k == k0[t]:
                    k = np.random.randint(0,m)
                
                if np.exp(h[t,k] - h[t,k0[t]]) > np.random.rand():
                    s[t,i1:i2],s[t,i1+k] = 0.,1.

        if irepeat%n == 0: print('irepeat:',irepeat) 

    return s

#=========================================================================================
def fit_multiplicative(s,n,m):   
    l = s.shape[0]
    i1tab,i2tab = itab(n,m) 

    y = s.copy()
    
    nloop = 20

    nm = n*m
    nm1 = nm - m
    
    w_infer = np.zeros((nm,nm))
    h0_infer = np.zeros(nm)

    wini = np.random.normal(0.0,1./np.sqrt(nm),size=(nm1,nm))
    h0ini = np.random.normal(0.0,1./np.sqrt(nm),size=nm)   
    for i in range(n):
        i1,i2 = i1tab[i],i2tab[i]
        x = np.hstack([s[:,:i1],s[:,i2:]])
        
        #-------------------------------------------------------------
        # covariance[ia,ib]
        cab_inv = np.empty((m,m,nm1,nm1))
        eps = np.empty((m,m,l))
        for ia in range(m):
            for ib in range(m):
                if ib != ia:
                    eps[ia,ib,:] = y[:,i1+ia] - y[:,i1+ib]

                    which_ab = eps[ia,ib,:] !=0.                    
                    xab = x[which_ab]          

                    # ----------------------------
                    xab_av = np.mean(xab,axis=0)
                    dxab = xab - xab_av
                    cab = np.cov(dxab,rowvar=False,bias=True)
                    cab_inv[ia,ib,:,:] = linalg.pinv(cab,rcond=1e-15)
        #-------------------------------------------------------------
        
        w = wini[:,i1:i2].copy()
        h0 = h0ini[i1:i2].copy()
        
        cost = np.full(nloop,100.) 
        for iloop in range(nloop):
            h = h0[np.newaxis,:] + np.dot(x,w)
            
            # stopping criterion --------------------
            p = np.exp(h)
            p_sum = p.sum(axis=1)
            p /= p_sum[:,np.newaxis]

            cost[iloop] = ((y[:,i1:i2] - p[:,:])**2).mean()
            if iloop > 1 and cost[iloop] >= cost[iloop-1]: break
            #-----------------------------------------
            
            for ia in range(m):
                wa = np.zeros(nm1)
                ha0 = 0.
                for ib in range(m):
                    if ib != ia:

                        which_ab = eps[ia,ib,:] !=0.

                        eps_ab = eps[ia,ib,which_ab]
                        xab = x[which_ab]

                        # ----------------------------
                        xab_av = xab.mean(axis=0)
                        dxab = xab - xab_av

                        h_ab = h[which_ab,ia] - h[which_ab,ib]                    
                        ha = np.divide(eps_ab*h_ab,np.tanh(h_ab/2.), out=np.zeros_like(h_ab), where=h_ab!=0)                        

                        dhdx = dxab*((ha - ha.mean())[:,np.newaxis])
                        dhdx_av = dhdx.mean(axis=0)

                        wab = cab_inv[ia,ib,:,:].dot(dhdx_av)   # wa - wb
                        h0ab = ha.mean() - xab_av.dot(wab)      # ha0 - hb0
                        
                        wa += wab
                        ha0 += h0ab
                        
                w[:,ia] = wa/m
                h0[ia] = ha0/m
                
        w_infer[:i1,i1:i2] = w[:i1,:]
        w_infer[i2:,i1:i2] = w[i1:,:]
        h0_infer[i1:i2] = h0

    return w_infer,h0_infer
