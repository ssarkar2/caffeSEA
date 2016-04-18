from train.trainModel import *
from eval.evalModel import *
import sys


if __name__ == '__main__':
    op = int(sys.argv[1])
    dataDir = '/scratch0/sem4/cmsc702/deepSEA/deepSEA_caffe/fullData/'
    if op == 0:
        trainToySEA()
    elif op == 1:
        trainFullSEA(dataDir, "\"../dumpModels/caffeSEAFull_\"")
        #trainFullSEA('C:\Sayantan\\acads\cmsc702\deepSEACaffe\\fullData\\')
    elif op == 2:
        evaluateModel(dataDir + 'Model/model_defn_new.prototxt' ,'/scratch0/sem4/cmsc702/deepSEA/deepSEA_caffe/dumpModels/caffeSEAFull__iter_30000.caffemodel', dataDir + 'test.mat', dataDir + 'hdf5FullInputDir/', 'runlog.txt')






