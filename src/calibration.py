import arrangeData as ar
import MVG 
import log_reg
import SVM
import GMM
from validator import compute_min_DCF

K=3

DTR, LTR = ar.load_data("..\Dataset-pulsar\Train.txt")
DTE, LTE = ar.load_data("..\Dataset-pulsar\Test.txt")
 
column = ar.mcol
row = ar.mrow 

def calibrate():

    #-------------------------------------------------------------------------
    #----------------------------------MVG------------------------------------
    #-------------------------------------------------------------------------

    options = {"m": None, #No PCA
               "gaussianization": "no",
               "K": K, 
               "pi": 0.5, 
               "costs": (1, 1)}
    gc = gaucl.GaussianClassifier("full covariance", "tied")
    v = xvalidator.CrossValidator(gc, D, L)
    min_DCF, scores, labels = v.kfold(options)
    #xvalidator.plot_bayes_error(scores, labels, "MVG")
    
    lr = logreg.LogRegClassifier(0, 0.5)
    v = xvalidator.CrossValidator(lr, row(scores), labels)
    min_DCF, scores, labels = v.kfold(options)
    xvalidator.plot_bayes_error(scores, labels, "MVGcalibrated")

