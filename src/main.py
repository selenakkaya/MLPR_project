import arrangeData
# import plotter 
import validator
import test_models
import PCA

import MVG
import log_reg
#----------------------------------------------------------------
#-------------------Load data------------------------------------
#----------------------------------------------------------------
"""
D, L = arrangeData.load_data("..\Dataset\Train.txt")


"""
DTE, LTE = arrangeData.load_data("..\Dataset\Test.txt")

#----------------------------------------------------------------
#-------------------Z_Normalization------------------------------
#----------------------------------------------------------------
"""
D_norm = arrangeData.z_norm(D) 
"""

#----------------------Load Data----------------------------------

D = arrangeData.DTR
L = arrangeData.LTR 
#D_z_norm, mu, sigma = arrangeData.z_norm(D)

#D_gauss = arrangeData.gaussianization_f(D)



#----------------------Show Heatmaps------------------------------

"""
plotter.show_heatmap(arrangeData.D, "Raw", "Greens")
plotter.show_heatmap(arrangeData.D[:, arrangeData.L==1], "Female", "Reds")
plotter.show_heatmap(arrangeData.D[:, arrangeData.L==0], "Male", "Blues")
"""





"""
#plot for raw feature
plotter.plt_RawFeature(D)

#plot for raw feature
plotter.plt_gaussianFeature(D)
"""
#----------------------Show PCA result------------------------------

"""

PCA.show_PCA_result()

"""


#----------------------------------------------------------------
#------------------------MVG-------------------------------------
#----------------------------------------------------------------

#-------------RAW Features, no PCA, K = 5------------------------
'''

def gaussian_classifiers(D, L):
    options = {"m": None, #No PCA
               "gaussianization": "no",
                "normalization" : "no",
                "K": 5, 
                "pi": 0.5, 
                "costs": (1, 1)}

    options["normalization"] = "no" 
    for options["gaussianization"] in ["no", "yes"]:
        for options["m"] in [None, 10, 9]:
            for options["pi"] in [0.5, 0.1, 0.9]:
                print(options)
                test_models.test_gauss_classifiers(D, L, options)

    options["normalization"] = "yes" 
    options["gaussianization"] ="no"
    for options["m"] in [None ,10, 9]:
        for options["pi"] in [0.5, 0.1, 0.9]:
            print(options)
            test_models.test_gauss_classifiers(D, L, options)

gaussian_classifiers(D, L)



'''

#-------------z-normed features, no PCA, K = 5------------------------
#gaussian_classifiers(D_z_norm, L)

"""
def gaussian_classifiers_PCA_11(D, L):
    options = {"m": None, #No PCA
               "gaussianization": "no",
               "K": 3, 
               "pi": 0.5, 
               "costs": (1, 1)}

    options["gaussianization"] = "no"
    for options["m"] in [11 ,10]:
        for options["pi"] in [0.5, 0.1, 0.9]:
            print(options)
            test_models.test_gauss_classifiers(D, L, options)


def gaussian_classifiers_gaussian_classifiers_with_gaussianization_PCA_11(D, L):
    options = {"m": None, #No PCA
               "gaussianization": "no",
               "K": 3, 
               "pi": 0.5, 
               "costs": (1, 1)}

    options["gaussianization"] = "yes"
    for options["m"] in [11 ,10]:
        for options["pi"] in [0.5, 0.1, 0.9]:
            print(options)
            test_models.test_gauss_classifiers(D, L, options)



def gaussian_classifiers_with_gaussianization(D, L):
    options = {"m": None, #No PCA
               "gaussianization": "no",
               "K": 3, 
               "pi": 0.5, 
               "costs": (1, 1)}

  
    options["gaussianization"] = "yes"
    for options["m"] in [None ,7, 6, 5]:
        for options["pi"] in [0.5, 0.1, 0.9]:
            print(options)
            test_models.test_gauss_classifiers(D, L, options)
"""

#----------------------------------------------------------------
#----------------------Logistic Regression-----------------------
#----------------------------------------------------------------


def logistic_regression(D, L):
    options = {"m": None,
               "gaussianization": "yes",
               "K": 5,
               "pT": 0.5,
               "pi": 0.5,
               "costs": (1, 1),
               "l": 0}
    
    for options["gaussianization"] in ["yes"]:
        print("")
        for options["pi"] in [0.5, 0.1, 0.9]:
            print("")
            for options["pT"] in [0.5, 0.1, 0.9]:
                print(options)
                test_models.test_logistic_regression(D, L, options)


logistic_regression(D,L)