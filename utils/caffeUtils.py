#this file contains functions that init/call/use caffe
import caffe
import numpy as np
import h5py

def initCaffe(solverList): #solverList is a list of 2-tuples (name, solverpath).
    caffe.set_mode_gpu()
    return [(solverName, caffe.get_solver(solverPath)) for (solverName, solverPath) in solverList]

def run_solvers(niter,solvers, disp_interval=10):
    blobs = ('loss',)# 'acc')
    (loss,) = ({name: np.zeros(niter) for name, _ in solvers} for _ in blobs)
    for it in range(niter):
        for name, s in solvers:
            s.step(1)  # run a single SGD step in Caffe
            (loss[name][it],) = (s.net.blobs[b].data.copy() for b in blobs)
        if it % disp_interval == 0 or it + 1 == niter:
            loss_disp = '; '.join('%s: loss=%.3f' %
                                  (n, loss[n][it])
                                  for n, _ in solvers)
            print '%3d) %s' % (it, loss_disp)
    # Save the learned weights from both nets.
    weight_dir = tempfile.mkdtemp()
    weights = {}
    for name, s in solvers:
        filename = 'weights.%s.caffemodel' % name
        weights[name] = os.path.join(weight_dir, filename)
        s.net.save(weights[name])
    return loss , weights


def forwardThroughNetwork(filename, net, dataLayerName, groundTruthName, outLayerName, batchsize):
    f = h5py.File(filename, 'r')
    numsamples = f[dataLayerName].shape[0]
    c = 0;
    flag = 1;
    while(1):
        c += 1;
        if c*batchsize <= numsamples:
            endidx = c*batchsize; currbatch = batchsize
        else:
            endidx = None; currbatch = numsamples-(c-1)*batchsize  #till the end
        if c%50 == 0:
            print 'processing batch', c, ' out of ', numsamples/batchsize
        net.blobs[dataLayerName].data[0:currbatch,:,:,:] = f[dataLayerName][batchsize*(c-1):endidx][:][:][:]
        out= net.forward()
        #print out.keys()
        #print out[groundTruthName].shape, out[outLayerName].shape
        if flag == 1:
            result = out[outLayerName]; ground = out[groundTruthName]; flag = 0
        else:
            result = np.vstack((result, out[outLayerName][0:currbatch][:])); ground = np.vstack((ground, out[groundTruthName][0:currbatch][:]));
        if endidx == None: break
    return [result, ground]


