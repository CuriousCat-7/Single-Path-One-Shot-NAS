import torch
import numpy as np
from typing import List


class UniformSampler(object):
    def __init__(self, num_choices:List[int]):
        self.idxs = [np.arange(num_choice) for num_choice in num_choices] # List[np.array]
    def random_choice(self):
        return [np.random.choice(idx).item() for idx in self.idxs]

    def state_dict(self)->dict:
        return dict()

    def load_state_dict(self, sd):
        return

    def __call__(self):
        return [self.random_choice()]


class MCUCBSampler(UniformSampler):
    """
    Monte Carlo UCB sampler
    """
    def __init__(self,
            num_choices:List[int],
            valid_iter,
            criterion,
            init_Q:float=0.0,
            c:float=1.0,
            reward:List[float]=[1.0, 0.0, -1.0], # TODO punish too big model
            alpha:float=0.99,
            ):
        super().__init__(num_choices)
        self.valid_iter = valid_iter
        self.criterion = criterion
        self.Q = [np.ones(num_choice)*init_Q for num_choice in num_choices] # List[np.array]
        self.N = [np.zeros(num_choice) for num_choice in num_choices] # List[np.array]
        self.t = 0
        self.c = c
        self.reward = reward
        self.alpha = alpha
        self.L = len(num_choices)  # number for layers
        self.archs = []

    @property
    def best_arch(self)-> List[int]:
        return [np.argmax(self.Q[i]).item() for i in range(self.L)]

    def ucb(self, archs:List[int], m:int) -> torch.Tensor:
        ucb_scores = []
        freqs = []
        values = []
        self.t += self.L * m
        for arch in archs:
            Q = 0
            N = 0
            for i in range(len(arch)):
                Q += self.Q[i][arch[i]]
                N += self.N[i][arch[i]]
            freq = self.c * np.sqrt( self.L*np.log(self.t/self.L)/N ) if N != 0 else np.float("inf")
            value = Q / self.L
            ucb_score = value + freq
            ucb_scores.append(ucb_score)
            freqs.append(freq)
            values.append(value)
        return torch.Tensor(ucb_scores), torch.Tensor(values), torch.Tensor(freqs)

    def update_N(self, archs:List[List[int]]):
        for arch in archs:
            for i in range(len(arch)):
                self.N[i][arch[i]] += 1

    def update_Q(self, archs:List[List[int]], reward:float):
        for arch in archs:
            for i in range(len(arch)):
                self.Q[i][arch[i]] = self.Q[i][arch[i]]*self.alpha + reward

    def state_dict(self)->dict:
        return dict(
                t=self.t,
                Q=self.Q,
                N=self.N,
                c=self.c,
                reward=self.reward,
                alpha=self.alpha)

    def load_state_dict(self, sd:dict):
        self.t = sd["t"]
        self.Q = sd["Q"]
        self.N = sd["N"]
        self.c = sd["c"]
        self.reward = sd["reward"]
        self.alpha = sd["alpha"]

    @torch.no_grad()
    def __call__(self, model, device, k=5, m=10, sample_num=100) -> List[List[int]]:
        archs = [self.random_choice() for i in range(sample_num)]
        # sample m archs based on ucb
        self.ucb_scores, self.values, self.freqs = self.ucb(archs, m)
        values, idxs = torch.topk(self.ucb_scores, k=m)
        idxs = idxs.tolist()
        m_archs = list(map(lambda i: archs[i], idxs))
        self.update_N(m_archs)
        # evaluate on validation set
        model.eval()
        losses = []
        inputs, targets = self.valid_iter.next()
        inputs, targets = inputs.to(device), targets.to(device)
        for arch in m_archs:
            outputs = model(inputs, arch)
            loss = self.criterion(outputs, targets)
            losses.append(loss.cpu().item())
        model.train()
        _, idxs = torch.topk(torch.FloatTensor(losses), k=k, largest=False)
        idxs = idxs.tolist()
        choosed_archs = list(map(lambda i: m_archs[i], idxs))
        """https://stackoverflow.com/a/11303241/7000846"""
        idxs = set(idxs) if  m > 100 else idxs
        not_choosed_archs = [arch for j, arch in enumerate(m_archs) if j not in idxs]
        self.update_Q(choosed_archs, self.reward[0])
        self.update_Q(not_choosed_archs, self.reward[1])
        #self.update_Q(punish_archs, self.reward[2])  # TODO
        return choosed_archs
