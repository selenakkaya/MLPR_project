
import numpy
import scipy
import scipy.optimize
import arrangeData

mcol = arrangeData.mcol
mrow = arrangeData.mrow 
    
def feature_exp(D):
    expansion = []
    for i in range(D.shape[1]):
        e = numpy.reshape(numpy.dot(mcol(D[:, i]), mcol(D[:, i]).T), (-1, 1), order='F')
        expansion.append(e)
    return numpy.vstack((numpy.hstack(expansion), D))

class LogRegClassifier:
    def __init__(self, l, pT, type):
        self.l = l
        self.pT = pT
        self.type = type
    
    def train(self, DTR, LTR):

        if type == 'quadratic':
            self.DTR = feature_exp(DTR) 
        else: 
            self.DTR = DTR

        self.LTR = LTR
        self.Z = LTR * 2.0 - 1.0
        self.M = DTR.shape[0]
        
        self.DTR0 = DTR[:, LTR==0]
        self.DTR1 = DTR[:, LTR==1]
        self.nF = self.DTR0.shape[1]
        self.nT = self.DTR1.shape[1]
        
        x0 = numpy.zeros(self.DTR.shape[0] + 1)
        v, J, d = scipy.optimize.fmin_l_bfgs_b(self._logreg_func, x0, approx_grad=True)
        self.w = v[0:-1]
        self.b = v[-1]
    
    def _logreg_func(self, v):
        w = v[0:self.M]
        b = v[-1]
        pT, l = self.pT, self.l
        
        S0 = numpy.dot(w.T, self.DTR0) + b
        S1 = numpy.dot(w.T, self.DTR1) + b
        crossEntropy = pT * numpy.logaddexp(0, -S1).mean()
        crossEntropy += (1-pT) * numpy.logaddexp(0, S0).mean()
        return crossEntropy + 0.5*l * numpy.linalg.norm(w)**2
    
    def validate(self, DTE, LTE):
        predictedLabels = self.classify(DTE)
        wrongPredictions = (LTE != predictedLabels).sum()
        samplesNumber = DTE.shape[1]
        errorRate = float(wrongPredictions / samplesNumber * 100)
        return wrongPredictions, errorRate
    
    def compute_scores(self, DTE):
        DTE = feature_exp(DTE) if self.type == 'quadratic' else DTE
        STE = numpy.dot(self.w.T, DTE) + self.b - numpy.log(self.nT/self.nF) #comment for score cal
        return STE
    
    