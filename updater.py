import chainer
from chainer import training, reporter, Variable
import chainer.functions as F


class ADDAUpdater(training.StandardUpdater):

    def __init__(self, iterator_source, iterator_target,
                 source_cnn, optimizer_target, optimizer_discriminator, args):
        iterators = {"main": iterator_source,
                     "target": iterator_target}
        optimizers = {"target_cnn": optimizer_target,
                      "discriminator": optimizer_discriminator}

        super().__init__(iterators, optimizers, device=args.device)

        self.source_cnn = source_cnn.encoder
        self.target_cnn = optimizer_target.target
        self.discriminator = optimizer_discriminator.target

        self.args = args

    def get_source(self):
        batch = next(self.get_iterator('main'))
        batch, labels = chainer.dataset.concat_examples(batch, device=self.args.device)
        return Variable(batch)

    def get_target(self):
        batch = next(self.get_iterator('target'))
        batch, labels = chainer.dataset.concat_examples(batch, device=self.args.device)
        return Variable(batch)

    def update_core(self):
        discriminator_optimizer = self.get_optimizer("discriminator")
        target_optimizer = self.get_optimizer("target_cnn")

        # get some batches (we are ignoring the labels)
        source_batch = self.get_source()
        target_batch = self.get_target()

        # update the discriminator
        source_encoding = self.source_cnn(source_batch)
        target_encoding = self.target_cnn(target_batch)

        D_source = self.discriminator(source_encoding)
        D_target = self.discriminator(target_encoding)

        # discriminator loss
        D_loss = - F.sum(F.log_softmax(D_source)[:, 0]) / self.args.batchsize \
                 - F.sum(F.log_softmax(D_target)[:, 1]) / self.args.batchsize

        # update discriminator
        self.discriminator.cleargrads()
        D_loss.backward()
        discriminator_optimizer.update()

        # now update the target CNN
        CNN_loss = -F.sum(F.log_softmax(D_target)[:, 0]) / self.args.batchsize

        self.target_cnn.cleargrads()
        CNN_loss.backward()
        target_optimizer.update()

        reporter.report({"loss/discrim": D_loss, "loss/encoder": CNN_loss})
