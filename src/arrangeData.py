import numpy


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
#---------------------z-normalize------------------------
#--------------------------------------------------------
def z_norm(D):
    mu_D = numpy.mean(D, axis=1) # mean of each column
    sigma_D = numpy.std(D, axis=1) #std dev of each column

    z_norm_D = (D - mcol(mu_D)) / mcol(sigma_D)
    return z_norm_D
#D_norm = z_norm(D) 


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