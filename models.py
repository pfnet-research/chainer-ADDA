import chainer
from chainer import links as L
from chainer import functions as F
from chainer.functions.loss import softmax_cross_entropy
from chainer.functions.evaluation import accuracy
from chainer import reporter

class Encoder(chainer.Chain):
	def __init__(self):
		super().__init__(
				cnn=L.VGG16Layers())

	def __call__(self, x):
		x = F.resize_image(x, output_shape=(224,224))
		encoding = self.cnn(x, layers=["fc7"])["fc7"]
		return encoding

class Classifier(chainer.Chain):
	def __init__(self, num_classes):
		super().__init__(
				l1=L.Linear(None, num_classes))

	def __call__(self, h):
		prediction = self.l1(h)
		return prediction

class Loss(chainer.Chain):
	def __init__(self, num_classes):
		super().__init__(
				encoder=Encoder(),
				classifier=Classifier(num_classes))

	def __call__(self, x, t):
		encode = self.encoder(x)
		classify = self.classifier(encode)
		self.accuracy = accuracy.accuracy(classify, t)
		self.loss = softmax_cross_entropy.softmax_cross_entropy(classify, t)
		reporter.report({"accuracy": self.accuracy, "loss": self.loss}, self)