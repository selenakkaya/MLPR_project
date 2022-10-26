import matplotlib.pyplot as plt

import arrangeData

Male_Raw = arrangeData.D[:,arrangeData.L == 0]
Female_Raw = arrangeData.D[:,arrangeData.L == 1]

def plt_RawFeature(D):
    
    plt.figure()
    for f in range(12):
        plt.hist(Male_Raw[f, :], color = "skyblue",bins=25, density=True, alpha=0.4)
        plt.hist(Female_Raw[f, :], bins=25,color = "red", density=True, alpha=0.4)
        plt.legend(['Male','Female'])
        plt.tight_layout()
        plt.xlabel("feature_"+str(f))
        i=0
        plt.savefig('rawDataPlots\hist_feature_'+str(i)+'_raw.jpeg')
        i=i+1
        plt.show()




