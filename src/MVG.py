import numpy as np

import arrangeData

mcol = arrangeData.mcol
mrow = arrangeData.mrow
D, L = arrangeData.load_data("..\Dataset\Train.txt")


class GaussianClassifier:
    def __init__(self, mode, tiedness, class_priors=None):
        if class_priors is None:
            class_priors = [0.5, 0.5]
        self.mode        = mode
        self.tiedness    = tiedness

    def train(self, DTR, LTR):
        D_M = DTR[:, LTR==0]
        D_F = DTR[:, LTR==1]
        
        mu_Interference = mcol(D_M.mean(1))
        mu_Pulsar = mcol(D_F.mean(1))
        
        
        C_M         = np.dot(D_M - mu_Interference, (D_M - mu_Interference).T) / D_M.shape[1]
        C_F         = np.dot(D_F - mu_Pulsar, (D_F - mu_Pulsar).T) / D_F.shape[1]
        naiveC_M    = C_M * np.eye(C_M.shape[0])
        naiveC_F    = C_F * np.eye(C_F.shape[0])
        tiedC      = (C_M*D_M.shape[1] + C_F*D_F.shape[1]) / float(DTR.shape[1])
        tiedNaiveC = (naiveC_M*D_M.shape[1] + naiveC_F*D_F.shape[1]) / float(DTR.shape[1])
        
        self.DTR, self.LTR         = DTR, LTR
        self.mu_Interference, self.mu_Pulsar = mu_Interference, mu_Pulsar
        self.C_M, self.C_F           = C_M, C_F
        self.naiveC_M, self.naiveC_F = naiveC_M, naiveC_F
        self.tiedC                 = tiedC
        self.tiedNaiveC            = tiedNaiveC


    def _logpdf_GAU_ND(self, X, mu, C):
        precision = np.linalg.inv(C)
        dimensions = X.shape[0]
        const = -0.5 * dimensions * np.log(2*np.pi)
        const += -0.5 * np.linalg.slogdet(C)[1]
    
        Y = []
        for i in range(X.shape[1]):
            x = X[:, i:i+1]
            res = const - 0.5 * np.dot((x-mu).T, np.dot(precision, (x-mu)))
            Y.append(res)
    
        return np.array(Y).ravel()
    
    def compute_lls(self, DTE):
        mu_Interference, mu_Pulsar    = self.mu_Interference, self.mu_Pulsar
        mode        = self.mode
        tiedness    = self.tiedness
        
        if mode == "full covariance" and tiedness == "untied":
            C_M, C_F = self.C_M, self.C_F
        elif mode == "naive bayes" and tiedness == "untied":
            C_M, C_F = self.naiveC_M, self.naiveC_F
        elif mode == "full covariance" and tiedness == "tied":
            C_M, C_F = self.tiedC, self.tiedC
        elif mode == "naive bayes" and tiedness == "tied":
            C_M, C_F = self.tiedNaiveC, self.tiedNaiveC
        else:
            print("ERROR: invalid ")
            quit()
            
        log_densities0 = self._logpdf_GAU_ND(DTE, mu_Interference, C_M)
        log_densities1 = self._logpdf_GAU_ND(DTE, mu_Pulsar, C_F)
        return log_densities0, log_densities1
    
    def compute_scores(self, DTE):
        log_densities0, log_densities1 = self.compute_lls(DTE)
        return log_densities1 - log_densities0

def ML_GAU(D):
    mu = mcol(D.mean(1))
    sigma = np.dot((D - mu), (D - mu).T) / D.shape[1]
    return mu, sigma


def logpdf_GAU_ND(D, mu, sigma):
    P = np.linalg.inv(sigma)
    C_F = 0.5 * D.shape[0] * np.log(2 * np.pi)
    c2 = 0.5 * np.linalg.slogdet(P)[1]
    c3 = 0.5 * (np.dot(P, (D - mu)) * (D - mu)).sum(0)
    return - C_F + c2 - c3

    
def compute_PCA(D,m):
    mu = mcol(D.mean(1))
    #covariance matrix
    C = np.dot((D - mu), (D - mu).T) / D.shape[1]
    #D.shape give us the number of value (n*m)
    s, U = np.linalg.eigh(C)
    U, s, Vh = np.linalg.svd(C)
    P = U[:,0:m]

    
    return P