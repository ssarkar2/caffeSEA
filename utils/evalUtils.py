import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

'''
test_np_array = np.random.random((5000, 919))
test_labels = np.random.randint(low = 0, high = 2, size = (5000, 919))
'''

# Alternate function definition in case the roc_curve function does not generate a good plot
# def generateROCplot(generatedLabels, actualLabels, stepSize = 0.01, startVal = 0.01, endVal = 0.99):


def generateROCplot(generatedLabels, actualLabels):
    index_vec = [1]*125 + [2]*690 + [3]*104
    auc_list = []
    for i in np.unique(index_vec):
        generatedLabelsClass = generatedLabels[:, index_vec == i]
        actualLabelsClass = actualLabels[:, index_vec == i]
        auc_vec = [0]*len(generatedLabelsClass[0])
        plt.figure()
        for j in range(0, len(generatedLabelsClass[0])):
            fpr, tpr, thresholds = roc_curve(actualLabelsClass[:,j], generatedLabelsClass[:,j])
            auc_vec[j] = auc(fpr, tpr)        
            plt.plot(fpr, tpr, 'k')
            plt.legend(loc='lower right')
            # plt.plot([0,1],[0,1],'r--')
            plt.xlim([-0.1,1.2])
            plt.ylim([-0.1,1.2])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.hold(True)
        plt.show()
        auc_list += [auc_vec]
    return auc_list
        
# auc_output = generateROCplot(test_np_array, test_labels)


# fileName should include location
def generateLossCurve(fileName):
    f = open(fileName)
    defText = f.read().split('\n')
    defText = defText[1:]
    x = []
    y = []    
    for line in defText:
        x += [int(line.split(')')[0])]
        y += [float(line.split(')')[1].split('=')[1])]
    f.close()
    plt.plot(x, y, 'k')
    plt.show()
