import chainer
from chainer import links as L
from chainer import functions as F
from chainer.functions.evaluation import accuracy
from chainer import reporter


class Encoder(chainer.Chain):

    def __init__(self, h=256, dropout=0.5):
        super(Encoder, self).__init__()
        self.dropout = dropout
        initialW = chainer.initializers.HeNormal()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 8, ksize=5, stride=1,
                                         initialW=initialW)
            self.conv2 = L.Convolution2D(8, 16, ksize=5, stride=1,
                                         initialW=initialW)
            self.conv3 = L.Convolution2D(16, 120, ksize=4, stride=1,
                                         initialW=initialW)
            self.fc4 = L.Linear(None, 500, initialW=initialW)

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2, 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2, 2)
        h = F.dropout(F.relu(self.conv3(h)), ratio=self.dropout)
        h = F.dropout(F.relu(self.fc4(h)), ratio=self.dropout)
        return h


class Discriminator(chainer.Chain):

    def __init__(self, h=500):
        super(Discriminator, self).__init__()
        initialW = chainer.initializers.HeNormal()
        with self.init_scope():
            self.l1 = L.Linear(None, h, initialW=initialW)
            self.l2 = L.Linear(h, h, initialW=initialW)
            self.l3 = L.Linear(h, 2, initialW=initialW)

    def __call__(self, x):
        l1 = F.leaky_relu(self.l1(x))
        l2 = F.leaky_relu(self.l2(l1))
        out = self.l3(l2)
        return out


class Classifier(chainer.Chain):

    def __init__(self, num_classes, dropout=0.5):
        super(Classifier, self).__init__()
        initialW = chainer.initializers.HeNormal()
        self.dropout = dropout
        with self.init_scope():
            self.l1 = L.Linear(None, num_classes, initialW=initialW)

    def __call__(self, x):
        prediction = self.l1(x)
        return prediction


class Loss(chainer.Chain):

    def __init__(self, num_classes):
        super(Loss, self).__init__()
        initialW = chainer.initializers.HeNormal()
        with self.init_scope():
            self.encoder = Encoder()
            self.classifier = Classifier(num_classes)

    def __call__(self, x, t):
        encode = self.encoder(x)
        classify = self.classifier(encode)

        self.accuracy = accuracy.accuracy(classify, t)
        self.loss = F.softmax_cross_entropy(classify, t)

        reporter.report({"accuracy": self.accuracy, "loss": self.loss}, self)
        return self.loss
