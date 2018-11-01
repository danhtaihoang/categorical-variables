import sys,os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as nplin
import scipy as sp
from scipy.linalg import pinvh as spinv
from scipy.sparse import csr_matrix,csc_matrix,random
from sklearn.preprocessing import OneHotEncoder as OneHotEncoder
from sklearn.linear_model import lasso_path
#import Bio

np.random.seed(10)
# =======================================================================================================================
"""
Function to generate sequences for testing. Generate a lot of sequences, then keep only sequences 
that are more likely `energetically'. Expect that this is more biologically realistic than
assuming that all possible sequences are part of the sample space.
"""
def zero_out(w,index,n_residues):
    w[index*n_residues:(index+1)*n_residues,index*n_residues:(index+1)*n_residues]=0.0

def split_seqs(seqs,index): #remove column i in the sequence 
    return np.copy(np.hstack([seqs[:,:index],seqs[:,index+1:]]))

def split_couplings(w,index,n_residues): #remove row block i in the coupling matrix when we're using only column i
    return np.copy(np.vstack([w[:index*n_residues],w[(index+1)*n_residues:]]))
print(split_couplings(np.arange(24).reshape((6,4)),1,2))

def nrgy(onehot,w,b):
    nrgy = onehot.multiply(onehot.dot(w) + b).toarray()
#    print(nrgy - np.log(2*np.cosh(nrgy)))
    return np.sum(nrgy - np.log(2*np.cosh(nrgy)),axis=1) #ln prob

def corr(onehot):
    #corr_out = onehot.transpose().dot(onehot).toarray()/np.float(onehot.shape[0])
    corr_out = np.cov(onehot.A,rowvar=False)
    expect = np.array(onehot.mean(axis=0)) #expectation value
    return expect,corr_out# - np.outer(expect,expect) #connected correlation function
    
def generate_sequences(n_residues,n_positions,n_seq):
    n_size = n_residues*n_positions
    n_trial = 10*(n_size) #monte carlo steps to find the right sequences
    res_interactions = np.sign(random(n_positions,n_positions,density=0.3).A)
    res_interactions = np.kron(res_interactions,np.ones((n_residues,n_residues)))
    w = res_interactions*(np.random.rand(n_size,n_size)-0.5)
    b = np.zeros((n_size))
    #different versions of random matrices
#    w = random(n_size,n_size,density=0.3).A -random(n_size,n_size,density=0.3).A
#    w /= np.sqrt(float(n_positions))
#    w = ((np.random.rand(n_size,n_size))-0.5)/np.sqrt(float(n_positions))

#    w = (np.random.normal(size=(n_size,n_size)))/(float(n_positions))
#    b = (np.random.rand(n_size)-0.5)/float(n_residues)
#    w = w+w.T #symmetric
    for indx in range(n_positions): #no terms W_iA,iB for B != A
        zero_out(w,indx,n_residues)
        #w[indx*n_residues:(indx+1)*n_residues,indx*n_residues:(indx+1)*n_residues]=0.0
#    trial_seq = np.random.randint(0,n_residues,size=(n_seq,n_positions)) #X(s,i)
    trial_seq = np.tile(np.random.randint(0,n_residues,size=(n_positions)),(n_seq,1))
    print(trial_seq[0])
    enc = OneHotEncoder(n_values=n_residues)
    onehot = csr_matrix(enc.fit_transform(trial_seq))
    old_nrgy = nrgy(onehot,w,b) + n_positions*(n_residues-1)*np.log(2)
    for trial in range(n_trial):
#        print('before',np.mean(old_nrgy))
        index_array = np.random.choice(range(n_positions),size=2,replace=False)
        index,index1 = index_array[0],index_array[1]
        r_trial = np.random.randint(0,n_residues,size=(n_seq))

        r_trial1 = np.random.randint(0,n_residues,size=(n_seq))
        
        mod_seq = np.copy(trial_seq)
        mod_seq[:,index] = r_trial

        mod_seq[:,index1] = r_trial1

        mod_nrgy = nrgy(csr_matrix(enc.fit_transform(mod_seq)),w,b) + n_positions*(n_residues-1)*np.log(2)
                
#        if trial%1000==0: print(seq_change)
        seq_change = mod_nrgy-old_nrgy > np.log(np.random.rand(n_seq))
        if trial>n_size:
            trial_seq[seq_change,index] = r_trial[seq_change]
            trial_seq[seq_change,index1] = r_trial1[seq_change]
            old_nrgy[seq_change] = mod_nrgy[seq_change]
        else:
            best_seq = np.argmax(mod_nrgy-old_nrgy)
            trial_seq = np.tile(mod_seq[best_seq],(n_seq,1))
            old_nrgy = np.tile(mod_nrgy[best_seq],(n_seq))
       
        if trial%n_size == 0: print('after',np.mean(old_nrgy))#,trial_seq[0:5])
    print(trial_seq[:10,:10])
    return trial_seq,w,b

#============================================================================================
# Now we want to set up the iteration for recovering w_true from the training sequences
# First, a function to calculate the correlation function from the 1-hot encoding

def iterate3(seqs,w,b,index,n_iter=6):
    #this version works, uses h_BC
    n_seq,n_pos = seqs.shape
    n_res = int(w.shape[0]/(n_pos))
    encoder = OneHotEncoder(n_values=n_res)
    onehot = csr_matrix(encoder.fit_transform(seqs)) 
    target = onehot.A[:,index*n_res:(index+1)*n_res]
    split_seqs0 = split_seqs(seqs,index)
    encoder = OneHotEncoder(n_values=n_res)
    onehot = csr_matrix(encoder.fit_transform(split_seqs0)) 

    b_init = np.mean(target,axis=0)
    b_init_zero = b_init > 0.99
    b_init[~b_init_zero] = 0.5/float(n_seq)
    b_init = np.log(b_init/np.sum(b_init))
    b[index*n_res:(index+1)*n_res] = b_init

    expect_onehot,corr_onehot = corr(onehot)
    corr_inv = spinv(corr_onehot)
    #print('corr_inv max',np.max(corr_inv))
    disc = np.zeros((n_iter))
    discr_old = 1e12
    for itera in range(n_iter):
        h_seq = np.array(onehot.dot(np.vstack([w[:index*n_res,index*n_res:(index+1)*n_res],\
     		w[(index+1)*n_res:,index*n_res:(index+1)*n_res]]))).reshape(n_seq,n_res)
        b[index*n_res:(index+1)*n_res] -= np.mean(b[index*n_res:(index+1)*n_res])
        h_seq += b[np.newaxis,index*n_res:(index+1)*n_res]
        #print('h_seq_max',np.max(h_seq))
        exp_h_seq = np.exp(h_seq)
        exp_h_seq /= np.sum(exp_h_seq,axis=1)[:,np.newaxis]
        disc[itera] = np.mean((target-exp_h_seq)**2)
        if itera>2 and disc[itera]>disc[itera-1]:
            w = w_old
            b = b_old
            break
        else:
            w_old = np.copy(w)
            b_old = np.copy(b)
        expect_h = np.mean(h_seq,axis=0)

        for res_index in range(n_res):
            w_temp = np.zeros((n_res*(n_pos-1)))
            for res_index1 in range(n_res):
                if res_index1 != res_index:

                    tgt = target[:,res_index]-target[:,res_index1]
                    which_BC = tgt != 0.0
                    tgt = tgt[which_BC]
                    n_BC = float(tgt.shape[0])
                    if n_BC>0:

                        h_BC = 0.5*(h_seq[which_BC,res_index] \
                               - h_seq[which_BC,res_index1])
                        ratio_BsumC = np.zeros(h_BC.shape)

                        expect_h_zero = h_BC != 0.0
                        ratio = 2*np.copy(tgt)
                        ratio[expect_h_zero] *= h_BC[expect_h_zero]
                                                /np.tanh(h_BC[expect_h_zero])
                        #print(res_index,res_index1,np.max(ratio))
                        ratio_BsumC += ratio/n_BC #so each sequence which is either B or C counts once
                        w_temp += onehot[which_BC,:].transpose().dot(ratio_BsumC \
                                                                -np.mean(ratio_BsumC))
            w_temp = corr_inv.dot(w_temp)/float(n_res)
            w[:index*n_res,index*n_res+res_index] = w_temp[:index*n_res]

            w[(index+1)*n_res:,index*n_res+res_index] = w_temp[(index)*n_res:]

            w[index*n_res:(index+1)*n_res,index*n_res+res_index] = 0.0

            b[index*n_res+res_index] = expect_h[res_index]#np.mean((ratio_BsumC))
            b[index*n_res+res_index] -= expect_onehot.dot(w_temp)
            
#        print(index,itera,disc[itera])
#        if np.sum(disc[itera-1,:]) < np.sum(disc[itera,:]): break

    return range(itera),disc[:itera]
#===========================================================================================
seqs,w_true,b_true = generate_sequences(5,100,20000)

w0 = (np.random.rand(w_true.shape[0],w_true.shape[1])-0.5)
b0 = np.random.rand(b_true.shape[0])-0.5
max_iter=60
n_seq,n_pos = seqs.shape
n_res = w0.shape[0]/n_pos

for index in range(n_pos):
    rang,disc = iterate3(seqs[:20000],w0,b0,index)
    w_tr_nonzero = np.abs(w_true[:,index*n_res:(index+1)*n_res]) <1

    w_nonzero = np.abs(w0[:,index*n_res:(index+1)*n_res]) <1

    w_true_nonzero = np.logical_and(w_tr_nonzero,w_nonzero)
    plt.plot((w0[:,index*n_res:(index+1)*n_res])[w_true_nonzero],\
			(w_true)[:,index*n_res:(index+1)*n_res][w_true_nonzero],'o',alpha = 0.2)

plt.show()
plt.close()
#===========================================================================================
fig, (ax1, ax2,ax3) = plt.subplots(1,3)
# make a little extra space between the subplots
fig.set_size_inches(14,44)
fig.subplots_adjust(hspace=0.5)

ax1.imshow(w_true, vmin=-.2,vmax=0.2)
ax1.set_xlabel('w_true')

ax2.imshow(w0,vmin=-.2,vmax=0.2)
ax2.set_xlabel('w_inferred')

ax3.imshow(w_true-w0,vmin=-.4,vmax=0.4)
ax3.set_xlabel('w_true-w_inferred')

plt.show()
plt.close()
