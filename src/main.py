import arrangeData
import plotter

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
D, L = arrangeData.load_data("..\Dataset\Train.txt")

#plot for raw feature
plotter.plt_RawFeature(D)