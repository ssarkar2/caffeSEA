import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import caffe
from utils.loadDataUtils import *
from utils.caffeUtils import *
import h5py

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
    defText = defText[1:-1]
    x = []
    y = []    
    for line in defText:
        x += [int(line.split(')')[0])]
        y += [float(line.split(')')[1].split('=')[1])]
    f.close()
    plt.plot(x, y, 'k')
    plt.show()


def evaluateModel(caffeProtoLoc, caffeModelLoc, testMatLoc, testHDF5Loc, runlogFile):
    generateLossCurve(runlogFile)

    testTxt = convertMatToHDF5(testMatLoc, testHDF5Loc, 0, 10000)
    f = open(testTxt, 'r')
    filenames = f.read().split('\n')
    caffe.set_mode_gpu() ; net = caffe.Net(caffeProtoLoc, caffe.TEST)
    result = []; ground = [];
    flag = 1; c = 0;
    for filename in filenames:
        c+=1
        if c == 4: break  #hack
        print 'processing', c, 'out of ', len(filenames), 'test hdf5 files'
        [r,g] = forwardThroughNetwork(filename, net, 'data', 'label', 'softmax', 64)
        if flag == 1:
            result = r; ground = g; flag = 0
        else:
            result = np.vstack((result, r)); ground = np.vstack((ground, g))

    auclist = generateROCplot(result, ground)
    print np.mean(auclist[0]), np.mean(auclist[1]), np.mean(auclist[2])  #if some answer is nan, it means one of the values in the list was nan.
    print [np.mean([j for j in auclist[i] if not np.isnan(j)]) for i in [0,1,2]]


def evaluateModelTorch(hdf5FileName):
    f = h5py.File(hdf5FileName, 'r')
    result = f['pred']
    ground = f['gt']

    auclist = generateROCplot(result, ground)
    print np.mean(auclist[0]), np.mean(auclist[1]), np.mean(auclist[2])  #if some answer is nan, it means one of the values in the list was nan.
    print [np.mean([j for j in auclist[i] if not np.isnan(j)]) for i in [0,1,2]]

    #0.904303375377 nan 0.839268867091
    #[0.90430337537739158, 0.930632247688994, 0.8392688670912748]

    #with prelu instead of relu
    #0.90129743803 nan 0.836661475715
    #[0.90129743802985118, 0.93133001457163556, 0.83666147571507732]




