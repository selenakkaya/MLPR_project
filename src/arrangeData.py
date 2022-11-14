import numpy


#--------------------------------------------------------
#-------------------convert matrices---------------------
#--------------------------------------------------------


def mcol(row): # convert a row array into a column one
    return row.reshape((row.size), 1)


def mrow(row): # convert a column array into a row one
    return row.reshape(1, row.size)

#--------------------------------------------------------
#--------------------Mean, Covariance--------------------
#--------------------------------------------------------

def empirical_mean(D):
    return mcol(D.mean(1))


def empirical_covariance(D, mu):
    n = numpy.shape(D)[1]
    DC = D - mcol(mu)
    C = 1 / n * numpy.dot(DC, numpy.transpose(DC))
    return C

    
#--------------------------------------------------------
#---------------------Loading Data-----------------------
#--------------------------------------------------------


def load_data(file):
    D = []
    L = []
    lines = open(file, "r")
    for l in lines:
        feature = mcol(numpy.array(l.split(",")[0:-1], dtype=float))
        label = l.split(",")[-1]

        L.append(label)
        D.append(feature)

    D = numpy.hstack(D)
    L = numpy.array(L, dtype=numpy.int32)
    return D, L

D, L = load_data("..\Dataset\Train.txt")

DTE, LTE = load_data("..\Dataset\Test.txt")

#--------------------------------------------------------
#---------------------z-normalize------------------------
#--------------------------------------------------------
def z_norm(D, mu=[], sigma=[]):
    if mu == [] or sigma == []:
        mu = numpy.mean(D, axis=1) # mean of each column
        sigma = numpy.std(D, axis=1) #std dev of each column

    z_norm_D = (D - mcol(mu)) / mcol(sigma)
    return z_norm_D, mu, sigma
#D_norm = z_norm(D) 

#--------------------------------------------------------
#--------------------Gaussianization---------------------
#--------------------------------------------------------

from scipy.stats import norm

def gaussianization_f(DTR, DTE=None):
    rankDTR = numpy.zeros(DTR.shape)
    for i in range(DTR.shape[0]):
        for j in range(DTR.shape[1]):
            rankDTR[i][j] = (DTR[i] < DTR[i][j]).sum()
    rankDTR = (rankDTR+1) / (DTR.shape[1]+2)
    
    if(DTE is not None):
        rankData_test = numpy.zeros(DTE.shape)
        for i in range(DTE.shape[0]):
            for j in range(DTE.shape[1]):
                rankData_test[i][j] = (DTR[i] < DTE[i][j]).sum()
        rankData_test = (rankData_test+1) / (DTR.shape[1]+2)
        return norm.ppf(rankDTR), norm.ppf(rankData_test)
    return norm.ppf(rankDTR)

D_gauss = gaussianization_f(D)

