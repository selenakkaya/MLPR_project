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
        
        mu_male = mcol(D_M.mean(1))
        mu_female = mcol(D_F.mean(1))

        DC0 = D_M - mu_male
        DC1 = D_F - mu_female
        Nc0 = D_M.shape[1]
        Nc1 = D_F.shape[1]
        
        
        C_M         = np.dot(DC0, DC0.T) / Nc0
        C_F         = np.dot(DC1, DC1.T) / Nc1
        naiveC_M    = C_M * np.eye(C_M.shape[0])
        naiveC_F    = C_F * np.eye(C_F.shape[0])
        tiedC      = (C_M * Nc0 + C_F*Nc1) / float(DTR.shape[1])
        tiedNaiveC = (naiveC_M * Nc0 + naiveC_F*Nc1) / float(DTR.shape[1])
        
        self.DTR, self.LTR         = DTR, LTR
        self.mu_male, self.mu_female = mu_male, mu_female
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
        mu_male, mu_female    = self.mu_male, self.mu_female
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
            
        log_densities0 = self._logpdf_GAU_ND(DTE, mu_male, C_M)
        log_densities1 = self._logpdf_GAU_ND(DTE, mu_female, C_F)
        return log_densities0, log_densities1
    
    def compute_scores(self, DTE):
        log_densities0, log_densities1 = self.compute_lls(DTE)
        return log_densities1 - log_densities0

