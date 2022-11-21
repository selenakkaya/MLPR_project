import matplotlib.pyplot as plt

import arrangeData
import numpy

import test_models
"""
D_gauss = arrangeData.gaussianization_f(arrangeData.DTR)

Male_Raw = arrangeData.DTR[:,arrangeData.LTR == 0]
Female_Raw = arrangeData.DTR[:,arrangeData.LTR == 1]

def plt_RawFeature(D):
    
    plt.figure()
    i=0
    for f in range(12):
        plt.hist(Male_Raw[f, :], color = "skyblue",bins=25, density=True, alpha=0.4)
        plt.hist(Female_Raw[f, :], bins=25,color = "red", density=True, alpha=0.4)
        plt.legend(['Male','Female'])
        plt.tight_layout()
        plt.xlabel("feature_"+str(f))
        plt.savefig('rawDataPlots\hist_feature_'+str(i)+'_raw.jpeg')
        i=i+1
        plt.show()


Male_gauss = arrangeData.D_gauss[:,arrangeData.L == 0]
Female_gauss = arrangeData.D_gauss[:,arrangeData.L == 1]
def plt_gaussianFeature(D):
    i=0
    plt.figure()
    for f in range(12):
        plt.hist(Male_gauss[f, :], color = "violet",bins=25, density=True, alpha=0.4)
        plt.hist(Female_gauss[f, :], bins=25,color = "darkorange", density=True, alpha=0.4)
        plt.legend(['Male','Female'])
        plt.tight_layout()
        plt.xlabel("feature_"+str(f))
        plt.savefig('gaussDataPlots\hist_feature_'+str(i)+'_gauss.jpeg')
        i=i+1
        plt.show()


def show_heatmap(D, title, color):
    plt.figure()
    pearson_matrix = numpy.corrcoef(D)
    plt.xlabel("Heatmaps of Pearson Correlation "+ title)
    plt.imshow(pearson_matrix, cmap=color, vmin=-1, vmax=1)
    plt.savefig("heatmaps\heatmap_%s.jpeg" % (title))

"""
def plot_lambda_minDCF(D, L):
    options = {"m": None,
               "gaussianization": "yes",
               "normalization" : "no",
               "K": 5,
               "pT": 0.5,
               "pi": 0.5,
               "costs": (1, 1),
               "l": 0}
    
    pis = [0.5, 0.1, 0.9]
    lambdas = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e1]
    min_DCFs = {pi: [] for pi in pis}

    
    options["normalization"] = "no"
    options["gaussianization"] = "no"
    for options["pi"] in pis:
        print("")
        for options["l"] in lambdas:
            print(options)
            min_DCF = test_models.test_logistic_regression(D, L, options)
            min_DCFs[options["pi"]].append(min_DCF)

    fn = "_RAW_"
    plt.figure()
    for pi in pis:
        plt.plot(lambdas, min_DCFs[pi], label='prior='+str(pi))
    plt.legend()
    plt.semilogx()
    plt.xlabel("λ")
    plt.ylabel("minDCF")
    plt.savefig("lambda-minDCF_Plots/lambda_minDCF" + fn + ".jpeg")

