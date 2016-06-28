from visualize_caffe import *
import sys

# Make sure caffe can be found
sys.path.append('../caffe/python/')

import caffe


# Load model
net = caffe.Net('/home/smistad/vessel_net/deploy.prototxt',
                '/home/smistad/vessel_net/snapshot_iter_3800.caffemodel',
                caffe.TEST)

visualize_weights(net, 'conv1', filename='conv1.png')
visualize_weights(net, 'conv2', filename='conv2.png')
