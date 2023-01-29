import arrangeData as ar
import MVG 
import log_reg
import SVM
import GMM
from validator import compute_min_DCF

def test():
    

    DTR, LTR = ar.load_data("..\Dataset\Train.txt")
    DTE, LTE = ar.load_data("..\Dataset\Test.txt")
    
    #DTR, mu, sigma = ar.z_norm(DTR)
    #DTE, mu, sigma = ar.z_norm(DTE, mu, sigma)

    print("MVG")
    for pi in [0.5, 0.1, 0.9]:
        gc = MVG.GaussianClassifier("full covariance", "tied")
        gc.train(DTR, LTR)
        scores = gc.compute_scores(DTE)
        min_DCF = compute_min_DCF(scores, LTE, pi, 1, 1)
        print(min_DCF)
    print("")

    
    print("LR")
    for pi in [0.5, 0.1, 0.9]:
        lr = log_reg.LogRegClassifier(0, 0.5, "quadratic") #Linear
        lr.train(DTR, LTR)
        scores = lr.compute_scores(DTE)
        min_DCF = compute_min_DCF(scores, LTE, pi, 1, 1)
        print(min_DCF)
    print("")


    print("SVM")
    for pi in [0.5, 0.1, 0.9]:
        s = SVM.SupportVectorMachines(1e-1, "Linear", 0.5, 1e-2, 1.0)
        s.train(DTR, LTR)
        scores = s.compute_scores(DTE)
        min_DCF = compute_min_DCF(scores, LTE, pi, 1, 1)
        print(min_DCF)
    print("")

    

    print("GMM")
    for pi in [0.5, 0.1, 0.9]:
        g = GMM.GMM_classifier(3, "full", "untied")
        g.train(DTR, LTR)
        scores = g.compute_scores(DTE)
        min_DCF = compute_min_DCF(scores, LTE, pi, 1, 1)
        print(min_DCF)
    print("")


test()