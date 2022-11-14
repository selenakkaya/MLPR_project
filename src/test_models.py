
import MVG
import validator

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


