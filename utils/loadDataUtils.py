#this file contains functions to read input data for training the network

import os
import scipy.io as sio
import h5py
import numpy as np
import urllib2
import tarfile


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
        swappedData = swapDims(matdata[matDataName])
        dataH5 = f.create_dataset('data', swappedData.shape, dtype='i1')  #i1 indicates 1byte sized integer.
        #print len(matdata[matDataName])
        #print len(matdata[matDataName][0])
        #print type(matdata[matDataName])
        dataH5[...] = swappedData  #matdata[matDataName] is (10000, 4, 1000). swappedData is (10000,1,4,1000)
        labelH5 = f.create_dataset('label', matdata[matLabelName].shape, dtype='i1') 
        labelH5[...] = matdata[matLabelName]  #(10000, 919)

    with open(('/').join([hdf5DataDir, fileName+'.txt' ]), 'w') as ftxt:
        ftxt.write(fullFileName)

def swapDims(nparr):
    shp = nparr.shape
    return np.reshape(nparr, (shp[0], 1, shp[1], shp[2]))
    
    
def createDir(dirname): #check if such a directory exists, else create a new one.
    try: os.stat(dirname)
    except: os.mkdir(dirname)
    

def getFileName(fullName):
    return fullName.split('/')[-1].split('.')[0]
    
def getDataNames(names):
    varnames = [i for i in names if '__' not in i]  #remove things like __global__, __version__ etc. now varnames is of size 2 only
    flag = (1, 0)['x' in varnames[0]]  #check if 'x' is present in the variable name
    return [varnames[flag], varnames[1-flag]]


def getFullTrainData(dataDir):
    url = 'http://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz'
    if (os.path.isfile(dataDir + 'train.mat') and os.path.isfile(dataDir + 'valid.mat') and os.path.isfile(dataDir + 'test.mat')):
        return
    else:
        file_name = dataDir + url.split('/')[-1]
        if not os.path.isfile(dataDir + 'deepsea_train_bundle.v0.9.tar.gz'):
            downloadFile(file_name, url)
        tar = tarfile.open(file_name, "r:gz")
        print 'extraction started...'
        extractFile(tar, 'train', dataDir); print 'train extract done'
        extractFile(tar, 'valid', dataDir); print 'valid extract done'
        extractFile(tar, 'test', dataDir); print 'test extract done'
        tar.close()
        os.remove(dataDir + 'deepsea_train_bundle.v0.9.tar.gz')
    
def extractFile(tar, fileName, outFolder):
    member = tar.getmember('deepsea_train/' + fileName + '.mat')
    member.name = os.path.basename(member.name)   
    tar.extract(member, outFolder)
    
def downloadFile(file_name, url):
    u = urllib2.urlopen(url)
    f = open(file_name, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print "Downloading: %s Bytes: %s" % (file_name, file_size)

    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        print status,

    f.close()

    
