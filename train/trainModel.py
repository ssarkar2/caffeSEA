from trainconfig import *
from utils.loadDataUtils import convertMatToHDF5
import sys
caffe_root = '../'  # path to caffe
sys.path.insert(0, caffe_root + 'python')
import caffe

def run_solvers(niter,solvers, disp_interval=10):
    blobs = ('loss', 'acc')
    loss, acc = ({name: np.zeros(niter) for name, _ in solvers}
                 for _ in blobs)
    for it in range(niter):
        for name, s in solvers:
            s.step(1)  # run a single SGD step in Caffe
            loss[name][it], acc[name][it] = (s.net.blobs[b].data.copy()
                                             for b in blobs)
        if it % disp_interval == 0 or it + 1 == niter:
            loss_disp = '; '.join('%s: loss=%.3f, acc=%2d%%' %
                                  (n, loss[n][it], np.round(100*acc[n][it]))
                                  for n, _ in solvers)
            print '%3d) %s' % (it, loss_disp)     
    # Save the learned weights from both nets.
    weight_dir = tempfile.mkdtemp()
    weights = {}
    for name, s in solvers:
        filename = 'weights.%s.caffemodel' % name
        weights[name] = os.path.join(weight_dir, filename)
        s.net.save(weights[name])
    return loss, acc, weights
	
def trainToySEA():
    cfg = getConfig()
    convertMatToHDF5(cfg.toyTrainData, cfg.hdf5ToyDir, 1)  #set readMode to 0, in case file already exists for faster execution
	caffe.set_mode_gpu()
	solver = caffe.get_solver("solver.prototxt")
	loss,acc,weights = run_solvers(100000,solver)
	del solver
    print 'hello'
        
def trainFullSEA():
    #download data from a file in the internet, or skip if the files already exist.
    pass