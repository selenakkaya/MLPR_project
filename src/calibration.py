import numpy as np
import pylab

import arrangeData as ar
import MVG 
import log_reg as LR
import SVM
import GMM
import validator

import matplotlib.pyplot as plt
"""
def bayesErrorPlot(dcf, mindcf, effPriorLogOdds, model):
    plt.figure()
    plt.plot(effPriorLogOdds, dcf, label='act DCF', color='r')
    plt.plot(effPriorLogOdds, mindcf, label='min DCF', color='b', linestyle="--")
    plt.xlim([min(effPriorLogOdds), max(effPriorLogOdds)])
    plt.legend([model + " - act DCF", model+" - min DCF"])
    plt.xlabel("prior log-odds")
    plt.ylabel("DCF")
    return
"""
def calibrate():
    validator.bayes_error_plot()

    K=3

    column = ar.mcol
    row = ar.mrow 

    
    D, L = ar.load_data("..\Dataset-pulsar\Train.txt")
    DTE, LTE = ar.load_data("..\Dataset-pulsar\Test.txt")

   

calibrate()