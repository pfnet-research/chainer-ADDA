import chainer
from chainer import links as L
from chainer import functions as F
from chainer.functions.loss import softmax_cross_entropy
from chainer.functions.evaluation import accuracy
from chainer import reporter
from matplotlib import pyplot as plt

from chainercv.visualizations import vis_image

class Encoder(chainer.Chain):
	def __init__(self, h=1024, dropout=0.5):
		self.dropout = dropout
		super().__init__(
				cnn=L.Convolution2D(None, 64, 3), #L.VGG16Layers(),
				l1=L.Linear(None, h)
				)

	def __call__(self, x):
		encoding = F.dropout(F.relu(self.cnn(x)), ratio=self.dropout) #, layers=["pool5"])["pool5"]
		encoding = F.dropout(F.relu(self.l1(encoding)), ratio=self.dropout)
		return encoding

class Discriminator(chainer.Chain):
	def __init__(self, h=500):
		super().__init__(
				l1=L.Linear(None, h),
				l2=L.Linear(h, h),
				l3=L.Linear(h, 1))
				
	def __call__(self, x):
		l1 = F.relu(self.l1(x))
		l2 = F.relu(self.l2(l1))
		out = self.l3(l2)
		return out

class Classifier(chainer.Chain):
	def __init__(self, num_classes):
		super().__init__(
				l1=L.Linear(None, num_classes))

	def __call__(self, x):
		prediction = self.l1(x)
		return prediction

class Loss(chainer.Chain):
	def __init__(self, num_classes):
		super().__init__(
				encoder=Encoder(),
				classifier=Classifier(num_classes))

	def __call__(self, x, t):
		#vis_image(x.data[1]*255.)
		#plt.show()
		encode = self.encoder(x)
		#print(encode.data)
		classify = self.classifier(encode)
		#print(classify.shape, t.shape)
		self.accuracy = accuracy.accuracy(classify, t)
		#print(t.shape)
		self.loss = F.softmax_cross_entropy(classify, t)
		reporter.report({"accuracy": self.accuracy, "loss": self.loss}, self)
		return self.loss