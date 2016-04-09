import caffe, numpy as np

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
