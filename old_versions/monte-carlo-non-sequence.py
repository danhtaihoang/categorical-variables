# 2018.11.07: equilibrium
def generate_sequences_vp_tai(w,n,m,l):
    nm = n*m
    nrepeat = 10*n
    nrelax = m
    
    b = np.zeros(nm)

    s0 = np.random.randint(0,m,size=(l,n)) # integer values    
    enc = OneHotEncoder(n_values=m)
    s = enc.fit_transform(s0).toarray()   
    
    e_old = np.sum(s*(s.dot(w.T)),axis=1)
    
    for irepeat in range(nrepeat):
        for i in range(n):
            
            for irelax in range(nrelax):            
                r_trial = np.random.randint(0,m,size=l)        
                s0_trial = s0.copy()
                s0_trial[:,i] = r_trial

                s = enc.fit_transform(s0_trial).toarray()                                    
                e_new = np.sum(s*(s.dot(w.T)),axis=1)

                t = np.exp(e_new - e_old) > np.random.rand(l)
                s0[t,i] = r_trial[t]
                e_old[t] = e_new[t]
      
        if irepeat%(5*n) == 0: print(irepeat,np.mean(e_old))

    return enc.fit_transform(s0).toarray()

#=========================================================================================
def generate_sequences_tai(w,n,m,l):
    i1tab,i2tab = itab(n,m)

    # initial s (categorical variables)
    s_ini = np.random.randint(0,m,size=(l,n)) # integer values
    #print(s_ini)

    # onehot encoder 
    enc = OneHotEncoder(n_values=m)
    s = enc.fit_transform(s_ini).toarray()
    #print(s)

    nrepeat = 500
    for irepeat in range(nrepeat):
        for i in range(n):
            i1,i2 = i1tab[i],i2tab[i]

            h = s.dot(w[i1:i2,:].T)              # h[t,i1:i2]
            
            h_old = (s[:,i1:i2]*h).sum(axis=1)   # h[t,i0]
            k = np.random.randint(0,m,size=l)
            for t in range(l):
                if np.exp(h[t,k[t]] - h_old[t]) > np.random.rand():
                    s[t,i1:i2] = 0.
                    s[t,i1+k[t]] = 1.
    return s   
