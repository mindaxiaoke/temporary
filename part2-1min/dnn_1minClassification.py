"""
Using Python Theano package:
http://deeplearning.net/software/theano/
Written by Minda Yang: minda.yang@columbia.edu
"""
import os
import sys
import inspect
import timeit
import numpy
import scipy.io
import theano
import theano.tensor as T
from itertools import product
import h5py

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None  #else theano.shared(value=activation(lin_output),name='output',borrow=True
            else activation(lin_output)   # else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

    def errors(self,y):
        """Return the mean squared root as the error
        :type y: theano.tensor.TensorType
        :param y: corresponds to a matrix that gives a vector of spectrum
                     for each example
        """
        return T.mean(T.square(self.output-y))

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.
            .. math::

                \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
                \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                    \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                      correct label

        Note: we use the mean instead of the sum so that
         the learning rate is less dependent on the batch size
        """
        #return -T.mean(T.log(self.output)[T.arange(y.shape[0]), y])
        return T.mean(T.square(self.output-y))

class data_mlp(object):
    """Multi-Layefr Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).

    This class is modified from MLP.
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out, n_hiddenLayers):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        :type n_hiddenLayers: int
        :param n_hiddenLayers: number of hidden layers

        """

        # The first hidden layer takes input from the "input", while the rest
        # receives input from the output of the previous layer.
        self.hiddenLayers = []
        for i in xrange(n_hiddenLayers):
            h_input = input if i == 0 else self.hiddenLayers[i-1].output
            h_in = n_in if i == 0 else n_hidden
            self.hiddenLayers.append(HiddenLayer(
                rng=rng,
                input=h_input,
                n_in=h_in,
                n_out=n_hidden,
                activation=T.tanh
            ))

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.outputLayer = LogisticRegression(
            input=self.hiddenLayers[-1].output,
            n_in=n_hidden,
            n_out=n_out
        )

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            sum([abs(x.W).sum() for x in self.hiddenLayers])
            + abs(self.outputLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            sum([(x.W ** 2).sum() for x in self.hiddenLayers])
            + (self.outputLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.outputLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.outputLayer.errors

        # the parameters of the model are the parameters of the multiple hidden
        # layers and the output layer  it is made out of
        self.params = sum([x.params for x in self.hiddenLayers], []) + \
            self.outputLayer.params


        # keep track of model input
        self.input = input


def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             batch_size=25, n_hidden=600, n_hiddenLayers=5, verbose=True,
             datasets=None):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type n_hiddenLayers: int
    :param n_hiddenLayers: number of hidden layers

    :type n_hiddenLayers: int
    :param n_hiddenLayers: number of hidden layers

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to

    :type dataset: None or list of theano.Tensor
    :param dataset: if dataset is None, use load_data to load the dataset

    """

    # load the dataset; download the dataset if it is not present
    if datasets is None:
        datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the data_mlp class
    classifier = data_mlp(
        rng=rng,
        input=x,
        n_in=32*32*3,
        n_hidden=n_hidden,
        n_hiddenLayers=n_hiddenLayers,
        n_out=10,
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                if verbose:
                    print(
                        'epoch %i, minibatch %i/%i, validation error %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            this_validation_loss * 100.
                        )
                    )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    if verbose:
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()


    print(('... optimization complete.\n... best validation score of %f %% '
           'obtained at iteration %i.\n... test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))


def LSUV_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=150,
             batch_size=15, n_hidden=20, n_hiddenLayers=4, verbose=True,
             dataset='ToTheano1min_5.mat'):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type n_hiddenLayers: int
    :param n_hiddenLayers: number of hidden layers

    :type n_hiddenLayers: int
    :param n_hiddenLayers: number of hidden layers

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to

    :type dataset: None or list of theano.Tensor
    :param dataset: if dataset is None, use load_data to load the dataset

    """

    # load the dataset; download the dataset if it is not present
    f = h5py.File(dataset,'r')
    train_set_x = f.get('train_x')
    train_set_x = numpy.array(train_set_x,dtype='float64')
    train_set_y = f.get('train_y')
    train_set_y = numpy.squeeze(numpy.array(train_set_y,dtype='float64'))
    valid_set_x = f.get('valid_x')
    valid_set_x = numpy.array(valid_set_x,dtype='float64')
    valid_set_y = f.get('valid_y')
    valid_set_y = numpy.squeeze(numpy.array(valid_set_y,dtype='float64'))
    test_set_x = f.get('test_x')
    test_set_x = numpy.array(test_set_x,dtype='float64')
    test_set_y = f.get('test_y')
    test_set_y = numpy.squeeze(numpy.array(test_set_y,dtype='float64'))
    train_set_x = theano.shared(train_set_x)
    train_set_y = T.cast(theano.shared(train_set_y),'int32')
    valid_set_x = theano.shared(valid_set_x)
    valid_set_y = T.cast(theano.shared(valid_set_y),'int32')
    test_set_x = theano.shared(test_set_x)
    test_set_y = T.cast(theano.shared(test_set_y),'int32')

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    print('Training batches: ' + str(n_train_batches))
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as matrix
    y = T.ivector('y')  # the labels are presented as 1D integer vector

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = data_mlp(
        rng=rng,
        input=x,
        n_in=33,
        n_hidden=n_hidden,
        n_hiddenLayers=n_hiddenLayers,
        n_out=5,
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ####################
    # This part written by Minda Yang
    # Code for initialization with LSUV
    # Idea from paper: All you need is a good init
    ####################
    margin = 0.05
    max_iter = 100
    needed_variance = 1.0
    # go through each hidden layer
    for i in xrange(n_hiddenLayers):
        # SVD decomposition to de-corelate
        w = classifier.hiddenLayers[i].W.eval()
        shape = w.shape
        flat_shape = (shape[0],numpy.prod(shape[1:]))
        a = numpy.random.standard_normal(flat_shape)
        u, _, v = numpy.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        classifier.hiddenLayers[i].W = q
        print('layer'+str(i), 'SVD Decompostion done.')
        # feed-forward once
        index = 0
        input = train_set_x[index * batch_size: (index + 1) * batch_size]
        cnt=0
        while cnt <= i:
            act = T.tanh(T.dot(input, classifier.hiddenLayers[cnt].W) + classifier.hiddenLayers[cnt].b)
            input = act
            cnt += 1
        var1 = numpy.var(act.eval())
        mean1 = numpy.mean(act.eval())
        print('layer'+str(i), 'var = '+ str(var1), 'mean = '+ str(mean1))
        iter_num = 0
        # feedforward mini-batch until convergence or maximum iteration
        while (abs(needed_variance - var1) > margin):
            weights = classifier.hiddenLayers[i].W
            classifier.hiddenLayers[i].W = weights / numpy.sqrt(var1)
            index = index+1
            input = train_set_x[index * batch_size: (index + 1) * batch_size]
            cnt = 0
            while cnt <= i:
                act = T.tanh(T.dot(input, classifier.hiddenLayers[cnt].W) + classifier.hiddenLayers[cnt].b)
                input = act
                cnt += 1
            var1 = numpy.var(act.eval())
            mean1 = numpy.mean(act.eval())
            print('layer'+str(i), 'var = '+ str(var1), 'mean = '+ str(mean1))
            iter_num += 1
            if iter_num > max_iter:
                print('Could not converge in ', iter_num, ' iterations, go to next layer')
                break

    ######## End of LSUV initialization #######

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    patience = 10*n_train_batches  # look as this many examples regardless
    patience_increase = 3  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.9999999  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                if verbose:
                    print(
                        'epoch %i, minibatch %i/%i, validation error %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            this_validation_loss * 100.
                        )
                    )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    if verbose:
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()


    print(('... optimization complete.\n... best validation score of %f %% '
           'obtained at iteration %i.\n... test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))

    return classifier

if __name__ == '__main__':
    print('start')
    classifier = LSUV_mlp() # pass a file to write the output
    scipy.io.savemat('dnnModel_1min_5_test1.mat', mdict={'tanhLayer0_W':classifier.hiddenLayers[0].W,
        'tanhLayer0_b':classifier.hiddenLayers[0].b.eval(),
        'tanhLayer1_W':classifier.hiddenLayers[1].W,
        'tanhLayer1_b':classifier.hiddenLayers[1].b.eval(),
        'tanhLayer2_W':classifier.hiddenLayers[2].W,
        'tanhLayer2_b':classifier.hiddenLayers[2].b.eval(),
        'tanhLayer3_W':classifier.hiddenLayers[3].W,
        'tanhLayer3_b':classifier.hiddenLayers[3].b.eval(),
        #'tanhLayer4_W':classifier.hiddenLayers[4].W,
        #'tanhLayer4_b':classifier.hiddenLayers[4].b.eval(),
        'outputLayer_W':classifier.outputLayer.W.eval(),
        'outputLayer_b':classifier.outputLayer.b.eval()})
