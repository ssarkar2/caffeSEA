from train.trainModel import *
from eval.evalModel import *
import sys


if __name__ == '__main__':
    op = int(sys.argv[1])
    dataDir = '/scratch0/sem4/cmsc702/deepSEA/deepSEA_caffe/fullData/'
    if op == 0:
        trainToySEA()
    elif op == 1:
        trainFullSEA(dataDir, "\"../dumpModels_1/caffeSEAFull_\"")
        #trainFullSEA('C:\Sayantan\\acads\cmsc702\deepSEACaffe\\fullData\\')
    elif op == 2:
        evaluateModel(dataDir + 'Model/model_defn_new.prototxt' ,'/scratch0/sem4/cmsc702/deepSEA/deepSEA_caffe/dumpModels_1/caffeSEAFull__iter_30000.caffemodel', dataDir + 'test.mat', dataDir + 'hdf5FullInputDir/', 'runlog.txt')


        evaluateModel(dataDir + 'Model/model_defn_new.prototxt' ,'/scratch0/sem4/cmsc702/deepSEA/deepSEA_caffe/dumpModels_1/caffeSEAFull__iter_44000.caffemodel', dataDir + 'test.mat', dataDir + 'hdf5FullInputDir/', 'runlog.txt')
    elif op == 3: #Torch test
        #hdf5FileName = '/scratch0/sem4/cmsc702/deepSEA/deepSEA_orig/DeepSEA/testmat.pred.h5' #predictions from model trained by us
        #hdf5FileName = '/scratch0/sem4/cmsc702/deepSEA/deepSEA_orig/DeepSEA/testmat_prelu.pred.h5' #predictions from model trained by us with modification (prelu instead of relu)
        hdf5FileName = '/scratch0/sem4/cmsc702/deepSEA/deepSEA_orig/DeepSEA/testmat_pretrainedmodel.pred.h5'  #predictions from pretrained model
        
        evaluateModelTorch(hdf5FileName)
    elif op == 4: #DanQ test
        #hdf5FileName = '/scratch0/sem4/cmsc702/danQ/DanQ/data/danq_pred_test.h5'  #modl trained by us
        hdf5FileName = '/scratch0/sem4/cmsc702/danQ/DanQ/data/danq_pred_test_githubmodel.h5'  #pretrained model
        ground = '../fullData/test.mat'
        evaluateModelDanQ(hdf5FileName, ground)
        






