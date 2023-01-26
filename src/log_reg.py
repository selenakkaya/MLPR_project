
import numpy
import scipy
import scipy.optimize
import arrangeData

mcol = arrangeData.mcol
mrow = arrangeData.mrow 
    
def feature_expansion(D):
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
            self.DTR = feature_expansion(DTR) 
        else: 
            self.DTR = DTR

        self.LTR = LTR
        self.Z = LTR * 2.0 - 1.0
        self.M = DTR.shape[0]
        
        self.DTR0 = DTR[:, LTR==0]
        self.DTR1 = DTR[:, LTR==1]

        
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
        cross_entropy = pT * numpy.logaddexp(0, -S1).mean()
        cross_entropy += (1-pT) * numpy.logaddexp(0, S0).mean()
        return cross_entropy + 0.5*l * numpy.linalg.norm(w)**2
    
    def validate(self, DTE, LTE):
        pred_labels = self.classify(DTE)
        wrongPredictions = (LTE != pred_labels).sum()
        sample_num = DTE.shape[1]
        errorRate = float(wrongPredictions / sample_num * 100)
        return wrongPredictions, errorRate
    
    def compute_scores(self, DTE):
        DTE = feature_expansion(DTE) if self.type == 'quadratic' else DTE
        STE = numpy.dot(self.w, DTE) + self.b - (self.LTR[self.LTR == 1].shape[0] / self.LTR[self.LTR == 0].shape[0]) 
        return STE
    
    