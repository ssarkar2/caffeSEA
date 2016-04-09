import os
import scipy.io as sio
import h5py


def convertMatToHDF5(matData, hdf5DataDir, readMode):
    createDir(hdf5DataDir)
    fileName = getFileName(matData) 
    fullFileName = ('/').join([hdf5DataDir, fileName + '.hdf5'])
    if readMode == 0:
        if os.path.isfile(fullFileName): return
    
    matdata = sio.loadmat(matData)
    [matDataName, matLabelName] = getDataNames(matdata.keys())
    #print getDataNames(matdata.keys())
    
    with h5py.File(fullFileName,'w') as f:  
        dataH5 = f.create_dataset('data', matdata[matDataName].shape, dtype='i1')  #i1 indicates 1byte sized integer. #chunking enabled
        dataH5[...] = matdata[matDataName]  #(10000, 4, 1000)
        labelH5 = f.create_dataset('label', matdata[matLabelName].shape, dtype='i1') 
        labelH5[...] = matdata[matLabelName]  #(10000, 919)

    with open(('/').join([hdf5DataDir, fileName+'.txt' ]), 'w') as ftxt:
        ftxt.write(fullFileName)

    
    
    
def createDir(dirname): #check if such a directory exists, else create a new one.
    try: os.stat(dirname)
    except: os.mkdir(dirname)
    

def getFileName(fullName):
    return fullName.split('/')[-1].split('.')[0]
    
def getDataNames(names):
    varnames = [i for i in names if '__' not in i]  #remove things like __global__, __version__ etc. now varnames is of size 2 only
    flag = (1, 0)['x' in varnames[0]]  #check if 'x' is present in the variable name
    return [varnames[flag], varnames[1-flag]]

    
