import chainer
from chainer import training, reporter, cuda, Variable
import chainer.functions as F

import numpy as np

class ADDAUpdater(training.StandardUpdater):

	def __init__(self, iterator_source, iterator_target, source_cnn, optimizer_target, optimizer_discriminator, args):
		
		iterators = {"main": iterator_source, "target": iterator_target}
		optimizers = {"target_cnn": optimizer_target, "discriminator": optimizer_discriminator}

		super().__init__(iterators, optimizers, device=args.device)

		self.source_cnn = source_cnn.encoder
		self.target_cnn = optimizer_target.target.encoder
		self.discriminator = optimizer_discriminator.target

		self.args = args

	def _convert_batch(self, x):
		#x = self.converter(x, self.args.device)
		batch = chainer.dataset.concat_examples(next(iter), device=device_id)
		return Variable(x[0])

	def get_source(self):
		batch, labels = chainer.dataset.concat_examples(next(self.get_iterator('main')), device=self.args.device)
		return Variable(batch)

	def get_target(self):
		batch, labels = chainer.dataset.concat_examples(next(self.get_iterator('target')), device=self.args.device)
		return Variable(batch)

	def update_core(self):
		discriminator_optimizer = self.get_optimizer("discriminator")
		target_optimizer = self.get_optimizer("target_cnn")
		
		# get some batches (we are ignoring the labels)
		source_batch = self.get_source()
		target_batch = self.get_target()

		# update the discriminator
		D_source = self.discriminator(self.source_cnn(source_batch))
		D_target = self.discriminator(self.target_cnn(target_batch))

		# discriminator loss
		D_loss = -F.sum(F.log(D_source)) / self.args.batchsize \
				 -F.sum(F.log(1.-D_target)) / self.args.batchsize

		# update discriminator
		self.discriminator.cleargrads()
		D_loss.backward()
		discriminator_optimizer.update()

		# now update the target CNN
		CNN_loss = -F.sum(F.log(D_target)) / self.args.batchsize
		self.target_cnn.cleargrads()
		CNN_loss.backward()
		target_optimizer.update()
