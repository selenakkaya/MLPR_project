
import validator

import MVG
import log_reg
import SVM
import GMM

#-------------------------------------------------------------------------------#
#--------------------------------TEST MODELS------------------------------------#
#-------------------------------------------------------------------------------#


#----------------------------------TEST MVG-------------------------------------#


def test_gauss_classifiers(D, L, options):    
    gc = MVG.GaussianClassifier("full covariance", "untied")
    v = validator.CrossValidator(gc, D, L)
    min_DCF, _, _ =v.kfold(options)
    print("Full covariance - untied: %.3f" % min_DCF)
    
    gc = MVG.GaussianClassifier("naive bayes", "untied")
    v = validator.CrossValidator(gc, D, L)
    min_DCF, _, _ =v.kfold(options)
    print("Naive bayes - untied: %.3f" % min_DCF)
    
    gc = MVG.GaussianClassifier("full covariance", "tied")
    v = validator.CrossValidator(gc, D, L)
    min_DCF, _, _ =v.kfold(options)
    print("Full covariance - tied: %.3f" % min_DCF)

    gc = MVG.GaussianClassifier("naive bayes", "tied")
    v = validator.CrossValidator(gc, D, L)
    min_DCF, _, _ =v.kfold(options)
    print("Naive Bayes - tied: %.3f" % min_DCF)
    print("")

#----------------------------------TEST LogReg-------------------------------------#

def test_logistic_regression(D, L, options):
    lr = log_reg.train_log_reg(D, L, options["l"], options["pT"], options["type"])
    v = validator.CrossValidator(lr, D, L)
    min_DCF, _, _ =v.kfold(options)
    print("Logistic regression: %.3f" % min_DCF)
    return min_DCF

#----------------------------------TEST SVM-------------------------------------#

def test_SVM(D, L, options):
    s = SVM.SupportVectorMachines(options["C"], options["mode"], options["pT"],  gamma=options["gamma"])
    v = validator.CrossValidator(s, D, L)
    min_DCF, _, _ =v.kfold(options)
    print("SVM: %.3f" % min_DCF)
    return min_DCF

#----------------------------------TEST GMM-------------------------------------#



def test_GMM(D, L, options):
    g = GMM.GMM_classifier(options["n"], options["mode"], options["tiedness"])
    v = validator.CrossValidator(g, D, L)
    min_DCF, _, _ =v.kfold(options)
    print("GMM: %.3f" % min_DCF)
    return min_DCF
