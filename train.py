import argparse
import os
from matplotlib import pyplot as plt
import copy

import chainer
from chainer import optimizers, serializers
from chainer.functions.evaluation import accuracy
from chainer.training import extensions
from chainer.datasets import tuple_dataset
from chainercv.datasets import TransformDataset
from chainercv.visualizations import vis_image
from chainercv.transforms import resize

from models import Loss, Discriminator
from updater import ADDAUpdater


def data2iterator(data, batchsize, multiprocess=False):
    train, test = data
    if multiprocess:
        train_iterator = chainer.iterators.MultiprocessIterator(train, batchsize)
        test_iterator = chainer.iterators.MultiprocessIterator(test, batchsize, shuffle=False, repeat=False)
    else:
        train_iterator = chainer.iterators.SerialIterator(train, batchsize)
        test_iterator = chainer.iterators.SerialIterator(test, batchsize, shuffle=False, repeat=False)
    return train_iterator, test_iterator


def pretrain_source_cnn(data, args, epochs=1000):
    print(":: pretraining source encoder")
    source_cnn = Loss(num_classes=10)
    if args.device >= 0:
        source_cnn.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(source_cnn)

    train_iterator, test_iterator = data2iterator(data, args.batchsize, multiprocess=False)

    # train_iterator = chainer.iterators.MultiprocessIterator(data, args.batchsize, n_processes=4)

    updater = chainer.training.StandardUpdater(iterator=train_iterator, optimizer=optimizer, device=args.device)
    trainer = chainer.training.Trainer(updater, (epochs, 'epoch') ,out=args.output)

    # learning rate decay
    # trainer.extend(extensions.ExponentialShift("alpha", rate=0.9, init=args.learning_rate, target=args.learning_rate*10E-5))

    trainer.extend(extensions.Evaluator(test_iterator, source_cnn, device=args.device))
    # trainer.extend(extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}'), trigger=(10, "epoch"))
    trainer.extend(extensions.snapshot_object(optimizer.target, "source_model_epoch_{.updater.epoch}"), trigger=(epochs, "epoch"))

    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.LogReport(trigger=(1, "epoch")))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    trainer.run()

    return source_cnn


def test_pretrained_on_target(source_cnn, target, args):
    print(":: testing pretrained source CNN on target domain")

    if args.device >= 0:
        source_cnn.to_gpu()

    with chainer.using_config('train', False):
        _, target_test_iterator = data2iterator(target, args.batchsize, multiprocess=False)

        mean_accuracy = 0.0
        n_batches = 0

        for batch in target_test_iterator:
            batch, labels = chainer.dataset.concat_examples(batch, device=args.device)
            encode = source_cnn.encoder(batch)
            classify = source_cnn.classifier(encode)
            acc = accuracy.accuracy(classify, labels)
            mean_accuracy += acc.data
            n_batches += 1
        mean_accuracy /= n_batches

        print(":: classifier trained on only source, evaluated on target: accuracy {}%".format(mean_accuracy))

def train_target_cnn(source, target, source_cnn, target_cnn, args, epochs=10000):
    print(":: training encoder with target domain")
    discriminator = Discriminator()

    if args.device >= 0:
        source_cnn.to_gpu()
        target_cnn.to_gpu()
        discriminator.to_gpu()

    # target_optimizer = chainer.optimizers.Adam(alpha=1.0E-5, beta1=0.5)
    target_optimizer = chainer.optimizers.RMSprop(lr=args.lr)
    # target_optimizer = chainer.optimizers.MomentumSGD(lr=1.0E-4, momentum=0.99)
    target_optimizer.setup(target_cnn.encoder)
    target_optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

    # discriminator_optimizer = chainer.optimizers.Adam(alpha=1.0E-5, beta1=0.5)
    discriminator_optimizer = chainer.optimizers.RMSprop(lr=args.lr)
    # discriminator_optimizer = chainer.optimizers.MomentumSGD(lr=1.0E-4, momentum=0.99)
    discriminator_optimizer.setup(discriminator)
    discriminator_optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

    source_train_iterator, source_test_iterator = data2iterator(source, args.batchsize, multiprocess=False)
    target_train_iterator, target_test_iterator = data2iterator(target, args.batchsize, multiprocess=False)

    updater = ADDAUpdater(source_train_iterator, target_train_iterator, source_cnn, target_optimizer, discriminator_optimizer, args)

    trainer = chainer.training.Trainer(updater, (epochs, 'epoch'), out=args.output)

    trainer.extend(extensions.Evaluator(target_test_iterator, target_cnn, device=args.device))
    # trainer.extend(extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}'), trigger=(10, "epoch"))
    trainer.extend(extensions.snapshot_object(target_cnn, "target_model_epoch_{.updater.epoch}"), trigger=(epochs, "epoch"))

    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.LogReport(trigger=(1, "epoch")))
    trainer.extend(extensions.PrintReport(
        ["epoch", "loss/discrim", "loss/encoder",
         "validation/main/loss", "validation/main/accuracy", "elapsed_time"]))

    trainer.run()


def main(args):
    # get datasets
    source_train, source_test = chainer.datasets.get_svhn()
    target_train, target_test = chainer.datasets.get_mnist(ndim=3, pad_channels=True)
    source = source_train, source_test

    # resize mnist to 32x32
    def transform(in_data):
        img, label = in_data
        img = resize(img, (32, 32))
        return img, label

    target_train = TransformDataset(target_train, transform)
    target_test = TransformDataset(target_test, transform)

    target = target_train, target_test

    # load pretrained source, or perform pretraining
    pretrained = os.path.join(args.output, args.pretrained_source)
    if not os.path.isfile(pretrained):
        source_cnn = pretrain_source_cnn(source, args)
    else:
        source_cnn = Loss(num_classes=10)
        serializers.load_npz(pretrained, source_cnn)

    # how well does this perform on target domain?
    test_pretrained_on_target(source_cnn, target, args)
    exit()
    # initialize the target cnn (do not use source_cnn.copy)
    target_cnn = Loss(num_classes=10)
    # copy parameters from source cnn to target cnn
    target_cnn.copyparams(source_cnn)

    train_target_cnn(source, target, source_cnn, target_cnn, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "-g", type=int, default=-1)
    parser.add_argument("--batchsize", "-b", type=int, default=256)
    parser.add_argument("--lr", "-lr", type=float, default=1.0E-4)
    parser.add_argument("--weight_decay", "-w", type=float, default=2.0E-5)
    parser.add_argument("--output", "-o", type=str, default="result")
    parser.add_argument("--pretrained_source", type=str,
                        default="source_model_epoch_1000")
    args = parser.parse_args()

    if args.device >= 0:
        chainer.cuda.get_device_from_id(args.device).use()

    main(args)
