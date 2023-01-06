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

    
    D, L = ar.load_data("..\Dataset\Test.txt")
    DTE, LTE = ar.load_data("..\Dataset\Test.txt")

    """

    options = {"m": None, #No PCA
               "gaussianization": "no",
               "normalization": "yes",
               "K": K, 
               "pi": 0.5, 
               "costs": (1, 1)}
    gc = MVG.GaussianClassifier("full covariance", "tied")
    v = validator.CrossValidator(gc, D, L)
    min_DCF, scores, labels = v.kfold(options)
    
    #plot bayes_error

    pis = np.linspace(-3, 3, 21)
    act_y = []
    min_y = []
 
    for p in pis:
        pi = 1.0 / (1.0 + np.exp(-p))
        min_y.append(validator.compute_min_DCF(scores, labels, pi, 1, 1))
        act_y.append(validator.compute_act_DCF(scores, labels, pi, 1, 1))

    def bayesError_plot_MVG(pis, min_y, act_y ):   
        pylab.title("MVG min DCF")
        pylab.plot(pis, min_y, color="c")
        pylab.plot(pis, act_y, color="m")
        pylab.ylim([0,3.1])
        pylab.xlim([-3,3])
        pylab.legend(['min_DCF Tied-cov','act_DCF Tied-cov',])
        pylab.xlabel("prior log-odds")
        pylab.ylabel("DCF")
        pylab.savefig('score_calibration_plots\MVG_BayError.jpeg')

    #bayesError_plot_MVG(pis, min_y, act_y )
    #validator.plot_ROC(scores, 'MVG_ROC_curve', labels, COLOR="y", show=False)

    

    #LR
    options = {"m": 10,
               "gaussianization": "no",
               "normalization": "yes",
               "K":K,
               "pT": 0.5,
               "pi": 0.5,
               "type": "linear",
               "costs": (1, 1),
               "l": 1e-06}
    lr = LR.LogRegClassifier(options["l"], options["pT"], options["type"])
    v = validator.CrossValidator(lr, D, L)
    min_DCF, scores1, labels1 = v.kfold(options)


  
    #plot bayes_error for LR

    pis = np.linspace(-3, 3, 21)
    act_y_LR = []
    min_y_LR = []
    
    for p in pis:
        pi = 1.0 / (1.0 + np.exp(-p))
        min_y_LR.append(validator.compute_min_DCF(scores1, labels1, pi, 1, 1))
        act_y_LR.append(validator.compute_act_DCF(scores1, labels1, pi, 1, 1))
   
    def bayesError_plot_LR(pis, min_y, act_y ):   
        pylab.title("LR min DCF")
        pylab.figure()
        pylab.plot(pis, min_y, color="c")
        pylab.plot(pis, act_y, color="m")
        pylab.ylim([0.1,0.8])
        pylab.xlim([-3,3])
        pylab.legend(['min_y','act_y'])
        pylab.xlabel("FPR")
        pylab.xlabel("TPR")

        pylab.savefig('score_calibration_plots\LR_BayError.jpeg')

    #bayesError_plot_LR(pis, min_y_LR, act_y_LR )
    #validator.plot_ROC(scores1, 'LR_ROC_curve', labels1, COLOR="c", show=False)
   

   
   



    

    #SVM
    options_l = {"m": None, "gaussianization": "no", "normalization" : "yes", "K": K,
               "k": 1.0, "pT": 0.5, "pi": 0.5, "costs": (1, 1),"mode": "Linear",
               "gamma": 1e-2, "C": 1.0} 

    options_q = {"m": None, "gaussianization": "no", "normalization" : "yes", "K": K,
               "k": 1.0, "pT": 0.5, "pi": 0.5, "costs": (1, 1),"mode": "Quadratic",
               "gamma": 1e-2, "C": 1.0}  

    options_rbf = {"m": None, "gaussianization": "no", "normalization" : "yes", "K": K,
               "k": 1.0, "pT": 0.5, "pi": 0.5, "costs": (1, 1),"mode": "RBF",
               "gamma": 1e-2, "C": 1.0}       

    svm_l = SVM.SupportVectorMachines(options_l["C"], options_l["mode"], options_l["pT"], gamma=options_l["gamma"], k=options_l["k"])
    svm_q =SVM.SupportVectorMachines(options_q["C"], options_q["mode"], options_q["pT"], gamma=options_q["gamma"], k=options_q["k"])
    svm_rbf = SVM.SupportVectorMachines(options_rbf["C"], "RBF", options_rbf["pT"], gamma=options_rbf["gamma"], k=options_rbf["k"])
    
    vl = validator.CrossValidator(svm_l, D, L)    
    vq = validator.CrossValidator(svm_q, D, L)
    vrbf = validator.CrossValidator(svm_rbf, D, L)

    min_DCF_l, scores_l, labels_l = vl.kfold(options_l)
    min_DCF_q, scores_q, labels_q = vq.kfold(options_q)
    min_DCF_rbf, scores_rbf, labels_rbf = vrbf.kfold(options_rbf)
    
    #plot bayes_error for SVM

    pis = np.linspace(-3, 3, 21)
    act_y_SVM_l = []
    min_y_SVM_l = []
    
    act_y_SVM_q = []
    min_y_SVM_q = []

    act_y_SVM_rbf = []
    min_y_SVM_rbf = []

    for p in pis:
        pi = 1.0 / (1.0 + np.exp(-p))
        min_y_SVM_l.append(validator.compute_min_DCF(scores_l, labels_l, pi, 1, 1))
        act_y_SVM_l.append(validator.compute_act_DCF(scores_l, labels_l, pi, 1, 1))

        min_y_SVM_q.append(validator.compute_min_DCF(scores_q, labels_q, pi, 1, 1))
        act_y_SVM_q.append(validator.compute_act_DCF(scores_q, labels_q, pi, 1, 1))

        min_y_SVM_rbf.append(validator.compute_min_DCF(scores_rbf, labels_rbf, pi, 1, 1))
        act_y_SVM_rbf.append(validator.compute_act_DCF(scores_rbf, labels_rbf, pi, 1, 1))


    def bayesError_plot_SVM(pis, min_y, act_y ):   

        pylab.title("SVM min DCF")
        pylab.plot(pis, min_y, color="c")
        pylab.plot(pis, act_y, color="y")
        pylab.ylim(0, 1.1)
        pylab.legend(['act_DCF','min_DCF'])
        pylab.xlabel("FPR")
        pylab.ylabel("TPR")

        pylab.savefig('score_calibration_plots\SVM_BayError.jpeg')
    
    bayesError_plot_SVM(pis, min_y_SVM_l, act_y_SVM_l )
    #validator.plot_ROC(scores_l, 'SVM_ROC_curve', labels_l, COLOR="y", show=False) #Linear
    #validator.plot_ROC(scores_q, 'SVM_ROC_curve', labels_q, COLOR="g", show=False) #Quadratic
    #validator.plot_ROC(scores_rbf, 'SVM_ROC_curve', labels_rbf, COLOR="m", show=False) #RBF


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



    #plot bayes_error for GMM Full Cov 8 Gau

    pis = np.linspace(-3, 3, 21)
    act_y_GMM = []
    min_y_GMM = []
    
    for p in pis:
        pi = 1.0 / (1.0 + np.exp(-p))
        min_y_GMM.append(validator.compute_min_DCF(scores3, labels3, pi, 1, 1))
        act_y_GMM.append(validator.compute_act_DCF(scores3, labels3, pi, 1, 1))

    def bayesError_plot_GMM(pis, min_y, act_y ):   

        pylab.title("GMM min DCF")
        pylab.figure()
        pylab.plot(pis, min_y, color="c")
        pylab.plot(pis, act_y, color="y")
        pylab.ylim(0, 1.1)
        pylab.legend(['act_DCF','min_DCF'])
        pylab.xlabel("FPR")
        pylab.ylabel("TPR")

        
        pylab.savefig('score_calibration_plots\GMM_BayError.jpeg')
    
    bayesError_plot_GMM(pis, min_y_GMM, act_y_GMM )
    #validator.plot_ROC(scores, 'GMM_ROC_curve', labels, COLOR="y", show=False)
calibrate()


