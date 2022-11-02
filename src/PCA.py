import numpy
import matplotlib.pyplot as plt

import arrangeData
import plotter



D, L = arrangeData.load_data("..\Dataset\Train.txt")


mu = D.mean(axis=1) # mean of all the columns of the dataset

#Subtract the mean to the dataset
D_centered = D-arrangeData.mcol(mu)

#Compute the covariance matrix C
N = numpy.shape(D)[1] 
C=numpy.dot(D_centered, D_centered.T) / N

s, U = numpy.linalg.eigh(C) # compute eigenvalues and eigenvectors of C



m10 = 10
m9 = 9

P10 = U[:, ::-1][:, 0:m10] 
y10 = numpy.dot(P10.T, D)
plotter.show_heatmap(y10, "raw_m_10", "Greens")
plotter.show_heatmap(y10[:, L==1], "raw_female_m_10", "Reds")
plotter.show_heatmap(y10[:, L==0], "raw_male_m_10", "Blues")


P9 = U[:, ::-1][:, 0:m9] 
y9 = numpy.dot(P9.T, D)
plotter.show_heatmap(y9, "raw_m_9", "Greens")
plotter.show_heatmap(y9[:, L==1], "raw_female_m_9", "Reds")
plotter.show_heatmap(y9[:, L==0], "raw_male_m_9", "Blues")



print('C = ',C)
print('mu = ',mu)





def PCA_reduce(D, m):
    mu = D.mean(axis=1) # mean of all the columns of the dataset
    D_centered = D-arrangeData.mcol(mu)
    #Compute the covariance matrix C
    N = numpy.shape(D)[1] 
    C=numpy.dot(D_centered, D_centered.T) / N

    s, U = numpy.linalg.eigh(C) # compute eigenvalues and eigenvectors of C

    P = U[:, ::-1][:, 0:m]


    PCA_D = numpy.dot(P.T, D)
    return PCA_D, P