from scipy import io
import numpy as np
import chainer
from chainer.datasets import tuple_dataset

from models import Loss

from matplotlib import pyplot as plt

def pretrain_source(args):
	source_cnn = Loss()
	if args.device >= 0:
		source_cnn.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(source_cnn)


def get_svhn(path, dtype=np.float32, label_dtype=np.int32):
	svhn = io.loadmat(path)

	images = svhn["X"].transpose(3,2,0,1)
	images = images.astype(dtype)
	images /= 255.

	labels = svhn["y"].astype(label_dtype)

	return tuple_dataset.TupleDataset(images, labels)

def main(args):
	#get datasets
	source = get_svhn("/mnt/sakuradata3/datasets/SVHN/train_32x32.mat")
	target = chainer.datasets.get_mnist(ndim=3)
	
	pretrain_source(source, args)
	
if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--device", "-g", type=int, default=-1)
	args = parser.parse_args()

	if args.device >= 0:	
	    chainer.cuda.get_device_from_id(args.device).use()

	main(args)