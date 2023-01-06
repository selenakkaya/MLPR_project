import numpy

import PCA
from arrangeData import z_norm, gaussianization_f

class CrossValidator:
    def __init__(self, classifier, D, L):
        self.D = D
        self.L = L
        self.classifier = classifier
        
    def kfold(self, options):
        D = self.D
        L = self.L
        classifier = self.classifier
        
        K = options["K"]
        m = options["m"]
        gaussianization = options["gaussianization"]
        normalization = options["normalization"]

        pi = options["pi"]
        (cfn, cfp) = options["costs"]
        
        samplesNumber = D.shape[1]
        N = int(samplesNumber / K)
        
        numpy.random.seed(seed=0)
        indexes = numpy.random.permutation(D.shape[1])
        
        scores = numpy.array([])
        labels = numpy.array([])

        for i in range(K):
            idxTest = indexes[i*N:(i+1)*N]
            
            idxTrainLeft = indexes[0:i*N]
            idxTrainRight = indexes[(i+1)*N:]
            idxTrain = numpy.hstack([idxTrainLeft, idxTrainRight])
            
            DTR = D[:, idxTrain]
            LTR = L[idxTrain]   
            DTE = D[:, idxTest]
            LTE = L[idxTest]
            
            #zed-normalizes the data with the mu and sigma computed with DTR
            if normalization == "yes":
                DTR, mu, sigma = z_norm(DTR)
                DTE, mu, sigma = z_norm(DTE, mu, sigma)
            
            if gaussianization == "yes":
                DTR, DTE = gaussianization_f(DTR, DTE)
            
            if m is not None: #PCA needed
                DTR, P = PCA.PCA_reduce(DTR, m)
                DTE = numpy.dot(P.T, DTE)
                
            classifier.train(DTR, LTR)
            scores_i = classifier.compute_scores(DTE)
            scores = numpy.append(scores, scores_i)
            labels = numpy.append(labels, LTE)
        min_DCF = compute_min_DCF(scores, labels, pi, cfn, cfp)
        return min_DCF, scores, labels


        
def assign_labels(scores, pi, cfn, cfp, threshold=None): #ok
    if threshold is None:
        threshold = -numpy.log(pi*cfn) + numpy.log((1-pi)*cfp)
    
    predictions = scores > threshold
    return numpy.int32(predictions)

def compute_FNR(CM): #ok
    return CM[0,1] / (CM[0,1]+CM[1,1])

def compute_FPR(CM): #ok
    return  CM[1,0] / (CM[0,0]+CM[1,0])

def compute_confusion_matrix(predictions, labels): #ok
    C = numpy.zeros((2, 2))
    C[0, 0] = ((predictions == 0) * (labels == 0)).sum()
    C[0, 1] = ((predictions == 0) * (labels == 1)).sum()
    C[1, 0] = ((predictions == 1) * (labels == 0)).sum()
    C[1, 1] = ((predictions == 1) * (labels == 1)).sum()
    return C

def compute_normalized_emp_bayes(CM, pi, cfn, cfp): #ok
    emp_bayes = compute_emp_bayes(CM, pi, cfn, cfp)
    return emp_bayes / min(pi*cfn, (1-pi)*cfp)


def compute_emp_bayes(CM, pi, cfn, cfp): #ok
    fnr = compute_FNR(CM)
    fpr = compute_FPR(CM)
    risk = pi*cfn*fnr + (1-pi)*cfp*fpr
    return risk


def compute_act_DCF(scores, labels, pi, cfn, cfp, threshold=None): #ok
    predictions = assign_labels(scores, pi, cfn, cfp, threshold=threshold)
    CM = compute_confusion_matrix(predictions, labels)
    return compute_normalized_emp_bayes(CM, pi, cfn, cfp)

def compute_min_DCF(scores, labels, pi, cfn, cfp): #ok
    thresholds = numpy.array(scores)
    thresholds.sort()
    
    numpy.concatenate([numpy.array([-numpy.inf]), thresholds, numpy.array([numpy.inf]) ])
    dcf_list = []
    for t in thresholds:
        dcf_list.append(compute_act_DCF(scores, labels, pi, cfn, cfp, threshold=t))
    return numpy.array(dcf_list).min()
    
def plot_ROC(scores, name, labels, COLOR, show):
    import pylab
    thresholds = numpy.array(scores)
    thresholds.sort()
    thresholds = numpy.concatenate([numpy.array([-numpy.inf]), thresholds, numpy.array([numpy.inf]) ])
    
    fpr = numpy.zeros(thresholds.size)
    tpr = numpy.zeros(thresholds.size)
    for id, t in enumerate(thresholds):
        predictions = numpy.int32(scores > t)
        CM = numpy.zeros((2,2))

        for i in range(2):
            for j in range(2):
                CM[i,j] =((predictions==i) * (labels==j)).sum()
        tpr[id] = CM[1, 1] / (CM[1, 1] + CM[0, 1])
        fpr[id] = CM[1, 0] / (CM[0, 0] + CM[1, 0])
    pylab.plot(fpr, tpr, color = COLOR)
    if show == True:    
        pylab.show()
    pylab.savefig('score_calibration_plots\_' +  name + '.jpeg')

