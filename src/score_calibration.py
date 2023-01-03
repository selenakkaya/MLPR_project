import numpy as np
import pylab

import arrangeData as ar
import MVG 
import log_reg as LR
import SVM
import GMM
import validator

import matplotlib.pyplot as plt

def calibrate():

    K=5

    column = ar.mcol
    row = ar.mrow 

    
    D, L = ar.load_data("..\Dataset-pulsar\Train.txt")
    DTE, LTE = ar.load_data("..\Dataset-pulsar\Test.txt")
    
    options = {"m": 10, #No PCA
               "gaussianization": "no",
               "normalization": "yes",
               "K": K, 
               "pi": 0.5, 
               "costs": (1, 1)}
    gc = MVG.GaussianClassifier("full covariance", "tied")
    v = validator.CrossValidator(gc, D, L)
    min_DCF, scores, labels = v.kfold(options)
    
    """
    #plot bayes_error

    pis = np.linspace(-3, 3, 21)
    act_y = []
    min_y = []
    
    for p in pis:
        pi = 1.0 / (1.0 + np.exp(-p))
        min_y.append(validator.compute_min_DCF(scores, labels, pi, 1, 1))
        act_y.append(validator.compute_act_DCF(scores, labels, pi, 1, 1))
    
    pylab.title("MVG min DCF")
    pylab.figure()
    pylab.plot(pis, min_y, color="c")
    pylab.plot(pis, act_y, color="m")
    pylab.ylim([0,3.1])
    pylab.xlim([-3,3])
    pylab.legend(['min_DCF Tied-cov','act_DCF Tied-cov',])
    pylab.xlabel("prior log-odds")
    pylab.ylabel("DCF")

    pylab.savefig('CalibrationPics\%s.jpeg' % 'MVG_min_y')
    """
    validator.plot_ROC(scores, labels, COLOR="y", show=False)

    

    #LR
    options = {"m": 10,
               "gaussianization": "no",
               "normalization": "yes",
               "K":K,
               "pT": 0.5,
               "pi": 0.5,
               "costs": (1, 1),
               "l": 1e-06}
    lr = LR.LogRegClassifier(options["l"], options["pT"])
    v = validator.CrossValidator(lr, D, L)
    min_DCF, scores1, labels1 = v.kfold(options)

    validator.plot_ROC(scores1, labels1, COLOR="r",show=False)

    """    
    #plot bayes_error for LR

    pis = np.linspace(-3, 3, 21)
    act_y_LR = []
    min_y_LR = []
    
    for p in pis:
        pi = 1.0 / (1.0 + np.exp(-p))
        min_y_LR.append(validator.compute_min_DCF(scores1, labels1, pi, 1, 1))
        act_y_LR.append(validator.compute_act_DCF(scores1, labels1, pi, 1, 1))

    pylab.title("LR min DCF")
    pylab.figure()
    pylab.plot(pis, min_y_LR, color="c")
    pylab.plot(pis, act_y_LR, color="m")
    pylab.ylim([0.1,0.8])
    pylab.xlim([-3,3])
    pylab.legend(['min_y','act_y'])
    pylab.xlabel("FPR")
    pylab.xlabel("TPR")

    pylab.savefig('CalibrationPics\%s.jpeg' % 'LR_min_y')


    
    #SVM
    options = {"m": None,
               "gaussianization": "no",
               "K": 3,
               "pT": 0.5,
               "pi": 0.5,
               "costs": (1, 1),
               "mode": "linear",
               "C": 1}
    svm = SVM.SupportVectorMachines(options["C"], options["mode"], options["pT"])
    v2 = validator.CrossValidator(svm, D, L)
    min_DCF, scores2, labels2 = v2.kfold(options)

    
    #plot bayes_error for SVM

    pis = np.linspace(-3, 3, 21)
    act_y_SVM = []
    min_y_SVM = []
    
    for p in pis:
        pi = 1.0 / (1.0 + np.exp(-p))
        min_y_SVM.append(validator.compute_min_DCF(scores2, labels2, pi, 1, 1))
        act_y_SVM.append(validator.compute_act_DCF(scores2, labels2, pi, 1, 1))
    pylab.title("SVM min DCF")
    pylab.figure()
    pylab.plot(pis, min_y_SVM, color="c")
    pylab.plot(pis, act_y_SVM, color="y")
    pylab.ylim(0, 1.1)
    pylab.legend(['act_DCF','min_DCF'])
    pylab.xlabel("FPR")
    pylab.ylabel("TPR")

    pylab.savefig('CalibrationPics\%s.jpeg' % 'SVM_min_y')



   """    
 
    
    #GMM
    options = {"m": 10,
               "gaussianization": "yes",
               "normalization" : "no",
               "K": K,
               "pi": 0.5,
               "costs": (1, 1),
               "mode": "full",
               "tiedness": "untied",
               "n": 3} 
    g = GMM.GMM_classifier(options["n"], options["mode"], options["tiedness"])
    v3 = validator.CrossValidator(g, D, L)
    min_DCF, scores3, labels3 = v3.kfold(options)
    validator.plot_ROC(scores3, labels3, COLOR="c", show=True)


    """
    #plot bayes_error for GMM Full Cov 8 Gau

    pis = np.linspace(-3, 3, 21)
    act_y_GMM = []
    min_y_GMM = []
    
    for p in pis:
        pi = 1.0 / (1.0 + np.exp(-p))
        min_y_GMM.append(validator.compute_min_DCF(scores3, labels3, pi, 1, 1))
        act_y_GMM.append(validator.compute_act_DCF(scores3, labels3, pi, 1, 1))
    pylab.title("GMM min DCF")
    pylab.figure()
    pylab.plot(pis, min_y_GMM, color="c")
    pylab.plot(pis, act_y_GMM, color="y")
    pylab.ylim(0, 1.1)
    pylab.legend(['act_DCF','min_DCF'])
    pylab.xlabel("FPR")
    pylab.ylabel("TPR")

    pylab.savefig('CalibrationPics\%s.jpeg' % 'GMM_min_y')
    """
calibrate()


