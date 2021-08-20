import torch

class priorityQ_torch(object):
    """Priority Q implelmentation in PyTorch

    Args:
        object ([torch.Tensor]): [The Queue to work on]
    """

    def __init__(self, val):
        self.q = torch.tensor([[val, 0]])
        # self.top = self.q[0]
        # self.isEmpty = self.q.shape[0] == 0

    def push(self, x):
        """Pushes x to q based on weightvalue in x. Maintains ascending order

        Args:
            q ([torch.Tensor]): [The tensor queue arranged in ascending order of weight value]
            x ([torch.Tensor]): [[index, weight] tensor to be inserted]

        Returns:
            [torch.Tensor]: [The queue tensor after correct insertion]
        """
        #if type(x) == np.ndarray:
        #    x = torch.tensor(x)
        if self.isEmpty():
            self.q = x
            self.q = torch.unsqueeze(self.q, dim=0)
            return
        idx = torch.searchsorted(self.q.T[1], x[1])
        #print(idx)
        self.q = torch.vstack([self.q[0:idx], x, self.q[idx:]]).contiguous()

    def top(self):
        """Returns the top element from the queue

        Returns:
            [torch.Tensor]: [top element]
        """
        return self.q[0]

    def pop(self):
        """pops(without return) the highest priority element with the minimum weight

        Args:
            q ([torch.Tensor]): [The tensor queue arranged in ascending order of weight value]

        Returns:
            [torch.Tensor]: [highest priority element]
        """
        if self.isEmpty():
            print("Can Not Pop")
        self.q = self.q[1:]

    def isEmpty(self):
        """Checks is the priority queue is empty

        Args:
            q ([torch.Tensor]): [The tensor queue arranged in ascending order of weight value]

        Returns:
            [Bool] : [Returns True is empty]
        """
        return self.q.shape[0] == 0


def dijkstra(adj):
    n = adj.shape[0]
    distance_matrix = torch.zeros([n, n])
    for i in range(n):
        u = torch.zeros(n, dtype=torch.bool)
        d = 1e16 * torch.ones(n)
        d[i] = 0
        q = priorityQ_torch(i)
        while not q.isEmpty():
            v, d_v = q.top()  # point and distance
            v = v.int()
            q.pop()
            if d_v != d[v]:
                continue
            for j, py in enumerate(adj[v]):
                if py == 0 and j != v:
                    continue
                else:
                    to = j
                    weight = py
                    if d[v] + py < d[to]:
                        d[to] = d[v] + py
                        q.push(torch.Tensor([to, d[to]]))
        distance_matrix[i] = d
    return distance_matrix, q.q


