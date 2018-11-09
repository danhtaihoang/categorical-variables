def fit_vp_update(s):
    l = s.shape[0]

    s_av = np.mean(s[:-1],axis=0)
    ds = s[:-1] - s_av
    c = np.cov(ds,rowvar=False,bias=True)
    #print(c)

    c_inv = linalg.pinv(c,rcond=1e-15)
    #print(c_inv)

    nm = n*m
    nloop = 100

    wini = np.random.normal(0.0,g/np.sqrt(nm),size=(nm,nm))

    w_infer = np.zeros((nm,nm))

    for i in range(n):
        i1,i2 = i1tab[i],i2tab[i]

        w = wini[i1:i2,:].copy()
        for iloop in range(nloop):
            h = np.dot(s[:-1],w.T)

            for ia in range(m):
                #dhds_av = np.zeros((m,nm))
                wa = np.zeros(nm)
                
                for ib in range(m):
                    if ib != ia:
                        # eps[t] = s[t+1,ia] - s[t+1,ib] 
                        eps = s[1:,i1+ia] - s[1:,i1+ib]

                        which_ab = eps!=0.

                        eps = eps[which_ab]

                        x = s[:-1]
                        sab = x[which_ab]
                        
                        # ----------------------------
                        sab_av = np.mean(sab,axis=0)
                        dsab = sab - sab_av
                        cab = np.cov(dsab,rowvar=False,bias=True)
                        cab_inv = linalg.pinv(cab,rcond=1e-15)
                        #print(c_inv)
                        
                        # ----------------------------
                        
                        h_ab = h[which_ab,ia] - h[which_ab,ib]

                        ha = eps*h_ab/np.tanh(h_ab/2.)

                        dhds = (ha - ha.mean())[:,np.newaxis]*dsab

                        dhds_av = dhds.mean(axis=0)
                        
                        wa += np.dot(cab_inv,dhds_av) # ???

                w[ia,:] = wa/m

        w_infer[i1:i2,:] = w     

    return w_infer
