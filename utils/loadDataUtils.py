#this file contains functions to read input data for training the network

import os
import scipy.io as sio
import h5py
import numpy as np
import urllib2
import tarfile
from train.trainconfig import *


def convertMatToHDF5(matData, hdf5DataDir, readMode, split=1000):  #split into files each containing 'split' number of samples
    createDir(hdf5DataDir)
    fileName = getFileName(matData)
    fullFileName = ('/').join([hdf5DataDir, fileName + '.hdf5'])   #check..remove later if not needed
    if readMode == 0:
        txtFile = ('/').join([hdf5DataDir, fileName + '.txt'])
        if os.path.isfile(txtFile): return txtFile

    print matData
    try:
        matdata = sio.loadmat(matData)
        [matDataName, matLabelName] = getDataNames(matdata.keys())
        with h5py.File(fullFileName,'w') as f:
            swappedData = swapDims(matdata[matDataName])
            dataH5 = f.create_dataset('data', swappedData.shape, dtype='i1')  #i1 indicates 1byte sized integer.
            dataH5[...] = swappedData  #matdata[matDataName] is (10000, 4, 1000). swappedData is (10000,1,4,1000)
            labelH5 = f.create_dataset('label', matdata[matLabelName].shape, dtype='i1')
            labelH5[...] = matdata[matLabelName]  #(10000, 919)


        matdata = sio.loadmat(matData)
        [matDataName, matLabelName] = getDataNames(matdata.keys())
        swappedData = swapDims(matdata[matDataName])
        datasize = swappedData.shape
        chunkCount = 0;
        for chunkNum in range(0,datasize[0]/split):
            chunkCount += 1
            with h5py.File(('/').join([hdf5DataDir, fileName + str(chunkCount) + '.hdf5']),'w') as f:
                dataH5 = f.create_dataset('data', (split, datasize[1], datasize[2], datasize[3]), dtype='i1')  #i1 indicates 1byte sized integer.
                dataH5[...] = swappedData[(chunkCount-1)*split : chunkCount*split][:][:][:]  #matdata[matDataName] is (10000, 4, 1000). swappedData is (10000,1,4,1000)
                labelH5 = f.create_dataset('label', (split, 919), dtype='i1')
                labelH5[...] = matdata[matLabelName][(chunkCount-1)*split : chunkCount*split][:]  #(10000, 919)




    except:
        print 'scipy loading of mat file failed. using h5py'
        #matDataName = callMatlabDimensionConverter(matData)
        callMatlabDimensionConverter(matData)
        with h5py.File(matData,'r') as hf:
            [matDataName, matLabelName] = getDataNames(hf.keys())
            with h5py.File(fullFileName,'w') as f:
                print hf[matDataName].shape   #(4000, 1, 1, 4400000)   #WRONG. CHECK
                dataH5 = f.create_dataset('data', hf[matDataName].shape, dtype='i1')  #i1 indicates 1byte sized integer.
                dataH5[...] = hf[matDataName]
                labelH5 = f.create_dataset('label', hf[matLabelName].shape, dtype='i1')
                labelH5[...] = hf[matLabelName]  #(10000, 919)

    with open(('/').join([hdf5DataDir, fileName+'.txt' ]), 'w') as ftxt:
        for i in range(chunkCount):
            ftxt.write(('/').join([hdf5DataDir, fileName + str(i+1) + '.hdf5\n']))

    return ('/').join([hdf5DataDir, fileName+'.txt' ])


def callMatlabDimensionConverter(matData):
    os.system(Config().matlabPath + ' -nodisplay -nosplash -nodesktop -r "addpath(\'./utils\'); matReshape(\'' + matData + '\'); quit"')
    #return matData.replace('.mat', '_swap.mat')


def swapDims(nparr):
    shp = nparr.shape
    return np.reshape(nparr, (shp[0], 1, shp[1], shp[2]))


def createDir(dirname): #check if such a directory exists, else create a new one.
    try: os.stat(dirname)
    except: os.mkdir(dirname)


def getFileName(fullName):
    #return fullName.split('/')[-1].split('.')[0]   #for ubuntu
    return fullName.split('\\')[-1].split('.')[0]   #for windows

def getDataNames(names):
    varnames = [i for i in names if '__' not in i]  #remove things like __global__, __version__ etc. now varnames is of size 2 only
    flag = (1, 0)['x' in varnames[0]]  #check if 'x' is present in the variable name
    return [varnames[flag], varnames[1-flag]]


def getFullTrainData(dataDir):
    url = 'http://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz'
    print dataDir + 'train.mat'
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
        #os.remove(dataDir + 'deepsea_train_bundle.v0.9.tar.gz')

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


