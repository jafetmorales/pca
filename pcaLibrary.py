#IMPLEMENTATION OF PCA FEATURE REDUCTION DOWN TO 95% VARIANCE BY JAFET MORALES
#INPUT IS THE "data" MATRIX
#each row is an feature or variable
#each column is a sample or observation
#OUTPUT IS THE "pc" MATRIX, WHICH WILL HAVE LESS ROWS THAN "data"
#ALL OTHER VARIABLES ARE FOR UNDERSTANDING WHAT IS HAPPENING
#YOU CAN UNCOMMENT THEM AND USE THEM FOR EXTRA FUNCTIONALITY,
#SUCH AS DENOISING

import numpy as np
from numpy import array
from numpy import diag
from scipy.linalg import svd

def cumsum(iterable):
    iterable= iter(iterable)
    s= iterable.next()
    yield s
    for c in iterable:
        s= s+ c
        yield s

def pcaFeatureReduction(data):
    #each row is an variable
    #each column is an observation
    u,s,vt=svd(np.cov(data))

    # sort the PCs by descending order of the singular values (i.e. by the
    # proportion of total variance they explain)
    ind = np.argsort(s)[::-1]
    uOrd = u[:,ind]
    sOrd = diag(s[ind])
    vtOrd = vt[ind,:]

    #FULL RECONSTRUCTION OF COVARIANCE MATRIX OF DATA
    #recFull=np.dot(uOrd,np.dot(sOrd,vtOrd))
    #print recFull

    coverage=list(cumsum(s))
    coverage=coverage/max(coverage);
    numComps=np.min(np.where(coverage>.95))+1;
    #95 percent variance in PCA components reconstruction OF COVARIANCE MATRIX OF DATA
    #recDenoised=np.dot(uOrd[:,:numComps],np.dot(sOrd[:numComps,:numComps],vtOrd[:numComps,:]))
    #print recDenoised

    #TRANSFORMATION OF THE DATA TO PCA DOMAIN
    #the maxtrix on the left side of USV is used. Each column is an eigenvector
    #so it has to be transposed to make the transformation
    trMat=np.transpose(uOrd[:,:numComps]);
    pc=np.dot(trMat,data)
    return pc

    #RECONSTRUCTION OR DENOISED DATA BACK IN TIME DOMAIN:
    #back=np.dot(uOrd[:,:numComps],pc)
    #print back
