import matplotlib.pyplot as plt

import arrangeData
import numpy

import eval

K=5

DTR, LTR = arrangeData.load_data("..\Dataset\Train.txt")

D_gauss = arrangeData.gaussianization_f(DTR)

Male_Raw = DTR[:,LTR == 0]
Female_Raw = DTR[:,LTR == 1]

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


Male_gauss = D_gauss[:,LTR == 0]
Female_gauss = D_gauss[:,LTR == 1]
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




############### For LR lambda-mindcf ###############


def plot_lambda_minDCF(D, L):
    options = {"m": None,
               "gaussianization": "no",
               "normalization" : "no",
               "type" : "linear",
               "K": K,
               "pT": 0.5,
               "pi": 0.5,
               "costs": (1, 1),
               "l": 1e-5}
    
    pis = [0.5, 0.1, 0.9]
    lambdas = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1e1, 1e2, 1e3]
    min_DCFs = {pi: [] for pi in pis}
    for options["pi"] in pis:
        print("")
        for options["l"] in lambdas:
            print(options)
            min_DCF = eval.test_logistic_regression(D, L, options)
            min_DCFs[options["pi"]].append(min_DCF)
    plt.figure()
    for pi in pis:
        plt.plot(lambdas, min_DCFs[pi], label='prior='+str(pi))
    plt.legend()
    plt.semilogx()
    plt.xlabel("λ")
    plt.ylabel("minDCF")
    plt.savefig("LR_plots\lambda_minDCF.jpeg")
############### For Linear SVM ###############

def plot_C_minDCF_L_SVM(D, L):
    options = {"m": None,
               "gaussianization": "yes",
               "normalization" : "no",
               "gamma" : 1,
               "K": K,
               "k":1.0,
               "pT": 0.5,
               "pi": 0.5,
               "costs": (1, 1),
               "mode": "Linear",
               "C": 1}
    
    pis = [0.5, 0.1, 0.9]
    C = [1e-4, 1e-3,  1e-2, 1e-1, 1]
    min_DCFs = {pi: [] for pi in pis}
    for options["pi"] in pis:
        print("")
        for options["C"] in C:
            print(options)
            min_DCF = eval.test_SVM(D, L, options)
            min_DCFs[options["pi"]].append(min_DCF)
    plt.figure()
    for pi in pis:
        plt.plot(C, min_DCFs[pi], label='prior='+str(pi))
    plt.legend()
    plt.semilogx()
    plt.xlabel("C")
    plt.ylabel("minDCF")
    #plt.savefig("C_minDCF_SVM_normalized.jpeg")    
    plt.savefig("C_minDCF_L_SVM_gau.jpeg")


############### For Quad SVM ###############

def plot_C_minDCF_Q_SVM(D, L):
    options = {"m": None,
               "gaussianization": "no",
               "normalization" : "yes",
               "gamma" : 1,
               "K": K,
               "k":1.0,
               "pT": 0.5,
               "pi": 0.5,
               "costs": (1, 1),
               "mode": "Quadratic",
               "C": 1}
    
    pis = [0.5, 0.1, 0.9]
    C = [1e-3,  1e-2, 1e-1, 1]
    min_DCFs = {pi: [] for pi in pis}
    for options["pi"] in pis:
        print("")
        for options["C"] in C:
            print(options)
            min_DCF = eval.test_SVM(D, L, options)
            min_DCFs[options["pi"]].append(min_DCF)
    plt.figure()
    for pi in pis:
        plt.plot(C, min_DCFs[pi], label='prior='+str(pi))
    plt.legend()
    plt.semilogx()
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.savefig("C_minDCF_Q_SVM_normalized.jpeg")    
    #plt.savefig("C_minDCF_SVM_gau.jpeg")

############### For RBF SVM ###############

def plot_minDCF_gamma_SVM(D, L):
    options = {"m": None,
               "gaussianization": "no",
               "normalization" : "yes",
               "gamma" : 1,
               "K": K,
               "k":1.0,
               "pT": 0.5,
               "pi": 0.5,
               "costs": (1, 1),
               "mode": "RBF",
               "C": 1}
    
    pis = [0.5, 0.1, 0.9]
    gamma = [1e-4, 1e-3, 1e-2, 1e-1, 1]
    min_DCFs = {pi: [] for pi in pis}
    for options["pi"] in pis:
        print("")
        for options["gamma"] in gamma:
            print(options)
            min_DCF = eval.test_SVM(D, L, options)
            min_DCFs[options["pi"]].append(min_DCF)
    plt.figure()
    for pi in pis:
        plt.plot(gamma, min_DCFs[pi], label='prior='+str(pi))
    plt.legend()
    plt.semilogx()
    plt.xlabel("gamma")
    plt.ylabel("minDCF")
    plt.savefig("gamma_minDCF_RBF_SVM_normalized.jpeg")    
    #plt.savefig("C_minDCF_SVM_gau.jpeg")

############### For GMM  ###############

def GMM_components_graph(D, L):
    options = {"m": None,
               "gaussianization": "no",
               "normalization" : "no",
               "K": K,
               "pi": 0.5,
               "costs": (1, 1),
               "mode": "full",
               "tiedness": "untied",
               "n": 1}
    
    ns = [1, 2]
    
    x_labels = ['2 components', '4 components']
    
    
    min_DCFs = {n: [] for n in ns}
    for options["n"] in ns:
        print("")
        print(options)
        min_DCF = eval.test_GMM(D, L, options)
        min_DCFs[options["n"]].append(min_DCF)
    plt.figure()
    for n in ns:
        # Create the bar plot
        plt.bar(range(len(ns)), ns)
    plt.xticks(range(len(ns)), x_labels)

    plt.semilogx()
    plt.xlabel("components")
    plt.ylabel(str(options["mode"]) + str(options["tiedness"]))
    plt.savefig("GMM_components_plot.jpeg")    
