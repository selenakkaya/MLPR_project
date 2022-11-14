import arrangeData
import plotter
import test_models
#----------------------------------------------------
#-------------------Load data------------------------
#----------------------------------------------------
"""
D, L = arrangeData.load_data("..\Dataset\Train.txt")

DTE, LTE = arrangeData.load_data("..\Dataset\Test.txt")

"""
#----------------------------------------------------
#-------------------Z_Normalization------------------
#----------------------------------------------------
"""
D_norm = arrangeData.z_norm(D) 
"""

#----------------------Load Data----------------------

D, L = arrangeData.load_data("..\Dataset\Train.txt")
D_gauss = arrangeData.gaussianization_f(D)



"""
#plot for raw feature
plotter.plt_RawFeature(D)

#plot for raw feature
plotter.plt_gaussianFeature(D)
"""
#----------------------------------------------------
#------------------------MVG-------------------------
#----------------------------------------------------


def gaussian_classifiers(D, L):
    options = {"m": None, #No PCA
               "gaussianization": "no",
               "K": 5, 
               "pi": 0.5, 
               "costs": (1, 1)}

    options["gaussianization"] = "no"
    for options["m"] in [None ,12]:
        for options["pi"] in [0.5, 0.1, 0.9]:
            print(options)
            test_models.test_gauss_classifiers(D, L, options)
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
#-----------------------invoke MVG --------------------------
gaussian_classifiers(D, L)
