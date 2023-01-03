import numpy

from PCA import PCA_reduce
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
                DTR, P = PCA_reduce(DTR, m)
                DTE = numpy.dot(P.T, DTE)
                
            classifier.train(DTR, LTR)
            scores_i = classifier.compute_scores(DTE)
            scores = numpy.append(scores, scores_i)
            labels = numpy.append(labels, LTE)
        min_DCF = compute_min_DCF(scores, labels, pi, cfn, cfp)
        return min_DCF, scores, labels


        
def assign_labels(scores, pi, cfn, cfp, threshold=None):
    if threshold is None:
        threshold = -numpy.log(pi*cfn) + numpy.log((1-pi)*cfp)
    
    predictions = scores > threshold
    return numpy.int32(predictions)

def compute_FNR(CM):
    return CM[0][1] /(CM[0][1] + CM[1][1] )

def compute_FPR(CM):
    return  CM[1][0] /(CM[0][0] + CM[1][0] )

def compute_confusion_matrix(predictions, labels):
    C = numpy.zeros((2, 2))
    C[0, 0] = ((predictions == 0) * (labels == 0)).sum()
    C[0, 1] = ((predictions == 0) * (labels == 1)).sum()
    C[1, 0] = ((predictions == 1) * (labels == 0)).sum()
    C[1, 1] = ((predictions == 1) * (labels == 1)).sum()
    return C

def compute_optimal_bayes_decision(loglikelihood_ratios, pi, cfn, cfp):
    threshold= - numpy.log((pi*cfn)/((1-pi)*cfp))
    return(1*(loglikelihood_ratios>threshold))


def compute_emp_bayes(CM, pi, cfn, cfp):
    fnr = CM[0, 1] / (CM[0, 1] + CM[1, 1])
    fpr = CM[1, 0] / (CM[0, 0] + CM[1, 0])
    risk = pi*cfn*fnr + (1-pi)*cfp*fpr
    return risk

def compute_normalized_emp_bayes(CM, pi, cfn, cfp):
    risk = compute_emp_bayes(CM, pi, cfn, cfp)
    return risk / min(pi*cfn, (1-pi)*cfp)

def compute_act_DCF(llrs, labels, pi, cfn, cfp):
    predicted_labels=compute_optimal_bayes_decision(llrs, pi, cfn, cfp)
    conf_matrix=compute_confusion_matrix(predicted_labels, labels)
    br= compute_emp_bayes(conf_matrix, pi, cfn, cfp) #bayes risk
    nbr= compute_normalized_emp_bayes(br, pi, cfn, cfp) # normalized bayes risk -> actual DCF

    return nbr

def compute_min_DCF(llrs, labels, pi, cfn, cfp):
    llrs_sorted= numpy.sort(llrs) #sorted logLikelihood ratios
    DCFs=[]
    FPRs=[]
    TPRs=[]

    for i in llrs_sorted:
        predicted_label=1*(llrs>t)
        conf_matrix=compute_confusion_matrix(predicted_label, labels)
        br= compute_emp_bayes(conf_matrix, pi, cfn, cfp)
        nbr= compute_normalized_emp_bayes(br, pi, cfn, cfp)
        DCFs.append(nbr)

        FPRs.append(compute_FPR(conf_matrix))
        TPRs.append(1-compute_FNR(conf_matrix))

    DCF_min =min(DCFs)

    index_t = DCFs.index(DCF_min)
    
    return (DCF_min, FPRs, TPRs, llrs_sorted[index_t])
    
