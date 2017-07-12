import argparse
import os
from scipy import io
import numpy as np
from matplotlib import pyplot as plt

import chainer
from chainer import optimizers, serializers
from chainer.training import extensions
from chainer.datasets import tuple_dataset
from chainercv.datasets import TransformDataset
from chainercv.visualizations import vis_image
from chainercv.transforms import resize
#from svhn import get_svhn

from models import Loss, Discriminator
from updater import ADDAUpdater

def data2iterator(data, batchsize):
    train, test = data
    train_iterator = chainer.iterators.SerialIterator(train, batchsize)
    test_iterator = chainer.iterators.SerialIterator(test, batchsize, shuffle=False, repeat=False)
    return train_iterator, test_iterator

def pretrain_source_cnn(data, args, epochs=100):
    source_cnn = Loss(num_classes=10)
    if args.device >= 0:
        source_cnn.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(source_cnn)

    train_iterator, test_iterator = data2iterator(data, args.batchsize)

    #train_iterator = chainer.iterators.MultiprocessIterator(data, args.batchsize, n_processes=4)

    updater = chainer.training.StandardUpdater(iterator=train_iterator, optimizer=optimizer, device=args.device)
    trainer = chainer.training.Trainer(updater, (epochs, 'epoch'))# ,out="")

    # learning rate decay
    #trainer.extend(extensions.ExponentialShift("alpha", rate=0.9, init=args.learning_rate, target=args.learning_rate*10E-5))

    trainer.extend(extensions.Evaluator(test_iterator, source_cnn, device=args.device))
    #trainer.extend(extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}'), trigger=(10, "epoch"))
    trainer.extend(extensions.snapshot_object(optimizer.target, "source_model_epoch_{.updater.epoch}"), trigger=(epochs, "epoch"))

    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.LogReport(trigger=(1, "epoch")))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    
    trainer.run()

    return source_cnn

def train_target_cnn(source, target, source_cnn, target_cnn, args, epochs=100):
    discriminator = Discriminator()

    if args.device >= 0:
        source_cnn.to_gpu()
        target_cnn.to_gpu()
        discriminator.to_gpu()

    target_optimizer = chainer.optimizers.Adam()
    target_optimizer.setup(target_cnn)

    discriminator_optimizer = chainer.optimizers.Adam()
    discriminator_optimizer.setup(discriminator)

    source_train_iterator, source_test_iterator = data2iterator(source, args.batchsize)
    target_train_iterator, target_test_iterator = data2iterator(target, args.batchsize)

    updater = ADDAUpdater(source_train_iterator, target_train_iterator, source_cnn, target_optimizer, discriminator_optimizer, args)

    trainer = chainer.training.Trainer(updater, (epochs, 'epoch')) # ,out="")

    # learning rate decay
    #trainer.extend(extensions.ExponentialShift("alpha", rate=0.9, init=args.learning_rate, target=args.learning_rate*10E-5))

    trainer.extend(extensions.Evaluator(target_test_iterator, target_cnn, device=args.device))
    #trainer.extend(extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}'), trigger=(10, "epoch"))
    trainer.extend(extensions.snapshot_object(target_cnn, "target_model_epoch_{.updater.epoch}"), trigger=(epochs, "epoch"))

    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.LogReport(trigger=(1, "epoch")))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    
    trainer.run()


def main(args):
    #get datasets
    source = chainer.datasets.get_svhn()
    target_train, target_test = chainer.datasets.get_mnist(ndim=3, pad_channels=True)

    # resize mnist to 32x32
    def transform(in_data):
        img, label = in_data
        img = resize(img, (32, 32))
        return img, label
    
    target_train = TransformDataset(target_train, transform)
    target_test = TransformDataset(target_test, transform)
    target = target_train, target_test

    # load pretrained source, or perform pretraining
    if not os.path.isfile(args.pretrained_source):
        source_cnn = pretrain_source_cnn(source, args)
    else:
        source_cnn = Loss(num_classes=10)
        serializers.load_npz(args.pretrained_source, source_cnn)

    # copy this for the target
    target_cnn = source_cnn.copy()

    train_target_cnn(source, target, source_cnn, target_cnn, args)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "-g", type=int, default=-1)
    parser.add_argument("--batchsize", "-b", type=int, default=128)
    parser.add_argument("--pretrained_source", type=str, default="source_model_epoch_100")
    args = parser.parse_args()

    if args.device >= 0:    
        chainer.cuda.get_device_from_id(args.device).use()

    main(args)