#this file contains functions to edit the standard prototexts

from models import caffe_pb2
from train.trainconfig import *
from google.protobuf.text_format import Merge

# new_data = {'max_iter': 5000, 'snapshot': 500}
def createSolverPrototxt(newData, saveLoc):
    f = open('models/toyModel/solverToy.prototxt')
    defText = f.read().split('\n')
    defText.remove('')
    solverDefaultDict = {module.split(': ')[0]: module.split(': ')[1] for module in defText}
    newDictionary = solverDefaultDict
    for key, value in newData.iteritems():
        newDictionary[key] = str(value)
    f.close()
    newSolverText = ['{}: {}'.format(key, value) for key, value in newDictionary.iteritems()]
    newSolverProto = '\n'.join(newSolverText)
    newFileName = saveLoc + 'solver_new.prototxt'
    fNew = open(newFileName, 'w')
    fNew.write(newSolverProto)
    fNew.close()
    return newFileName


def createModelPrototxt(outDir, trainTxt, testTxt = ''):
     net = caffe_pb2.NetParameter()
     modelDefn = Merge((open(Config().toyModelDefn,'r').read()), net)

     if trainTxt != '':
         modelDefn.layer[0].hdf5_data_param.source = trainTxt
     if testTxt != '':
         modelDefn.layer[1].hdf5_data_param.source = testTxt
     outputModelProtoLoc = outDir + 'model_defn_new.prototxt'
     open(outputModelProtoLoc, 'w').write(str(modelDefn))
     return outputModelProtoLoc
