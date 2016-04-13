#this file contains functions to edit the standard prototexts

from models import caffe_pb2

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
