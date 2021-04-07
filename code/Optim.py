import math
import torch.optim as optim

class Optim(object):

    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr, weight_decay = self.weight_decay)
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,milestones=[50,75],gamma=0.1)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr, weight_decay = self.weight_decay)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr, weight_decay = self.weight_decay)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay = self.weight_decay)
            #self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,milestones=[56,70],gamma=0.1)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, params, method, lr, max_grad_norm, lr_decay=1, start_decay_at=None, weight_decay = 0., milestones=[]):
        self.params = list(params)  # careful: params may be a generator
        self.last_ppl = None
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False
        self.weight_decay = weight_decay;
        self.milestones = list(milestones)
        self.decay_num = 0

        self._makeOptimizer()

    def step(self):
        # Compute gradients norm.
        #print("grad norm is step!!!!!!!!!!!!!!!!!!!!!!")
        grad_norm = 0
        for param in self.params:
            grad_norm += math.pow(param.grad.data.norm(), 2)

        grad_norm = math.sqrt(grad_norm)
      
        if grad_norm > 0:
            shrinkage = self.max_grad_norm / grad_norm
        else:
            shrinkage = 1.

        for param in self.params:
            if shrinkage < 1:
                param.grad.data.mul_(shrinkage)

        self.optimizer.step()
        return grad_norm

    # decay learning rate if val perf does not improve or we hit the start_decay_at limit
    def updateLearningRate(self, ppl, epoch):
        print("***"*20)
        #if self.start_decay_at is not None:
        if self.decay_num<len(self.milestones):
            if(epoch >= self.milestones[self.decay_num]):
                self.start_decay = True
                self.decay_num += 1
            else:
                self.start_decay = False
        #if self.last_ppl is not None and ppl > self.last_ppl:
            #self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)
        #only decay for one epoch
        self.start_decay = False

        self.last_ppl = ppl

        self._makeOptimizer()
