"""
@Description: 接收神经网路（模型）和优化器
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-25 10:08:21
"""
from timeit import default_timer
import numpy as np
from common.util import clip_grads
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0

    def fit(self, x, t, max_epoch: int = 10,
            batch_size: int = 32, max_grad: float = None,
            eval_interval: int = 20):
        """_summary_

        Parameters
        ----------
        x : _type_
            _description_
        t : _type_
            _description_
        max_epoch : int, optional
            _description_, by default 10
        batch_size : int, optional
            _description_, by default 32
        max_grad : float, optional
            梯度的最大范数。当梯度的范数超过这个值时，缩小梯度（梯度裁剪），解决长期依赖的问题, by default None
        eval_interval : int, optional
            _description_, by default 20
        """
        data_size = len(x)
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_cnt = 0

        start_time = default_timer()
        for _ in range(max_epoch):
            idx = np.random.permutation(np.arange(data_size))
            x = x[idx]
            t = t[idx]
            for iters in range(max_iters):
                batch_x = x[iters * batch_size:(iters + 1) * batch_size]
                batch_t = t[iters * batch_size:(iters + 1) * batch_size]
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = model.params, model.grads
                # params, grads = remove_duplicate(model.params, model.grads)
                # if max_grad is not None:
                #     clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_cnt += 1
                if (eval_interval is not None) and ((iters + 1) % eval_interval) == 0:
                    avg_loss = total_loss / loss_cnt
                    elapsed_time = default_timer() - start_time
                    print(
                        f'| epoch {(self.current_epoch+1):04d}| iter {iters+1}/{max_iters} | time {elapsed_time:.2f}s | loss {avg_loss:.2f}')
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_cnt = 0, 0
            self.current_epoch += 1

    def plot(self, ylim=None):
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('iterations (x' + str(self.eval_interval) + ')')
        plt.ylabel('loss')
        plt.show()


def remove_duplicate():
    pass
