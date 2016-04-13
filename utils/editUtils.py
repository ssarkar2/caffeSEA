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
    newProto = '\n'.join(newSolverText)
    newFileName = saveLoc + 'solver_new.prototxt'
    fNew = open(newFileName, 'w')
    fNew.write(newProto)
    fNew.close()


def createModelPrototxt(outDir):
     net = caffe_pb2.NetParameter()
     modelDefn = Merge((open(Config().toyModelDefn,'r').read()), net)
     print modelDefn
     print modelDefn.layer
     print modelDefn.name
     print modelDefn.layer[0]
     print modelDefn.layer[0].name
     print modelDefn.layer[0].hdf5_data_param
     print modelDefn.layer[0].hdf5_data_param.source
     modelDefn.layer[0].hdf5_data_param.source = 'hellothere'
     open(outDir + 'model_defn_new.prototxt', 'w').write(str(modelDefn))
