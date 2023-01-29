import numpy as np


import arrangeData
import plotter 
import validator
import eval
import PCA

import MVG
import log_reg
import SVM
import GMM

#----------------------------------------------------------------
#-------------------Load data------------------------------------
#----------------------------------------------------------------
K=5
D, L = arrangeData.load_data("..\Dataset\Train.txt")
DTE, LTE = arrangeData.load_data("..\Dataset\Test.txt")

#----------------------------------------------------------------
#-------------------Z_Normalization------------------------------
#----------------------------------------------------------------



#----------------------Load Data----------------------------------

D_z_norm, mu, sigma = arrangeData.z_norm(D)
D_gauss = arrangeData.gaussianization_f(D)



#----------------------Show Heatmaps------------------------------


plotter.show_heatmap(D, "Raw", "Greens")
plotter.show_heatmap(D[:,L==1], "Female", "Reds")
plotter.show_heatmap(D[:,L==0], "Male", "Blues")





#plot for raw feature
plotter.plt_RawFeature(D)

#plot for raw feature
plotter.plt_gaussianFeature(D)
#----------------------Show PCA result------------------------------



print("PCA RESULTS:")
PCA.show_PCA_result()




#----------------------------------------------------------------
#------------------------------MVG-------------------------------
#----------------------------------------------------------------

#-------------RAW Features, no PCA, K = 5------------------------

def gaussian_classifiers(D, L):
    options = {"m": None, #No PCA
               "gaussianization": "no",
                "normalization" : "no",
                "K": K, 
                "pi": 0.5, 
                "costs": (1, 1)}

    options["normalization"] = "no" 
    for options["gaussianization"] in ["no", "yes"]:
        for options["m"] in [None, 11, 10]:
            for options["pi"] in [0.5, 0.1, 0.9]:
                print(options)
                eval.test_gauss_classifiers(D, L, options)

    options["normalization"] = "yes" 
    options["gaussianization"] ="no"
    for options["m"] in [None, 11,10]:
        for options["pi"] in [0.5, 0.1, 0.9]:
            print(options)
            eval.test_gauss_classifiers(D, L, options)

gaussian_classifiers(D, L)





#-------------z-normed features, no PCA, K = 5------------------------
#gaussian_classifiers(D_z_norm, L)


def gaussian_classifiers_PCA_11(D, L):
    options = {"m": None, #No PCA
               "gaussianization": "no",
               "K": K, 
               "pi": 0.5, 
               "costs": (1, 1)}

    options["gaussianization"] = "no"
    for options["m"] in [11 ,10]:
        for options["pi"] in [0.5, 0.1, 0.9]:
            print(options)
            eval.test_gauss_classifiers(D, L, options)


def gaussian_classifiers_gaussian_classifiers_with_gaussianization_PCA_11(D, L):
    options = {"m": None, #No PCA
               "gaussianization": "no",
               "K": K, 
               "pi": 0.5, 
               "costs": (1, 1)}

    options["gaussianization"] = "yes"
    for options["m"] in [11 ,10]:
        for options["pi"] in [0.5, 0.1, 0.9]:
            print(options)
            eval.test_gauss_classifiers(D, L, options)






#----------------------------------------------------------------
#----------------------Logistic Regression-----------------------
#----------------------------------------------------------------

def logistic_regression(D, L):
    options = {"m": None,
               "gaussianization": "no",
               "normalization" : "no",
               "type" : "quadratic", #linear, quadratic
               "K": K,
               "pT": 0.5,
               "pi": 0.5,
               "costs": (1, 1),
               "l": 1e-5}
    
    for options["gaussianization"] in [ "no", "yes"]:
            for options["pi"] in [0.5, 0.1, 0.9]:
                print("")
                for options["pT"] in [0.5, 0.1, 0.9]:
                    print(options)
                    eval.test_logistic_regression(D, L, options)


logistic_regression(D,L)

#-------------------plot lambda - minDCF ----------------------------------------------#
#plotter.plot_lambda_minDCF(D, L)
plotter.plot_lambda_minDCF_quad(D, L)

#----------------------------------------------------------------
#-----------------------------SVM--------------------------------
#----------------------------------------------------------------

def SVM(D, L):
    options = {"m": None,
               "gaussianization": "no",
               "normalization" : "yes",
               "K": K,#fold
               "k": 1.0,
               "C":1e-1,
               "pT": 0.5,
               "pi": 0.5,
               "costs": (1, 1),
               "mode": "Linear", #Linear, Quadratic, RBF
               "gamma": 1e-2}
    for options["mode"] in ["RBF", "Quadratic",  "Linear"]:            
        for options["k"] in [1.0, 0.1]:
            for options["C"] in [1.0, 1e-1, 10.0]:
                for options["pi"] in [0.5, 0.1, 0.9]:
                    print("")
                    print(options)
                    eval.test_SVM(D, L, options)

SVM(D, L)
plotter.plot_C_minDCF_L_SVM(D, L) #change pT =0.1, 0.9 and take the plots
#plotter.plot_C_minDCF_Q_SVM(D, L) #change pT =0.1, 0.9 and take the plots
#plotter.plot_minDCF_gamma_SVM(D, L)
#----------------------------------------------------------------
#-----------------------------GMM--------------------------------
#----------------------------------------------------------------


def GMM(D, L):
    options = {"m": 10,
               "gaussianization": "yes",
               "normalization" : "no",
               "K": K,
               "pi": 0.5,
               "costs": (1, 1),
               "mode": "full",  #naive, full
               "tiedness": "untied", #tied, untied
               "n": 1}
    for options["n"] in [1, 2, 3]:
        for options["mode"] in [ "naive", "full"]:
            for options["tiedness"] in ["tied", "untied"]:
                print(options)
                eval.test_GMM(D, L, options)

GMM(D,L)
plotter.GMM_components_graph(D,L)


