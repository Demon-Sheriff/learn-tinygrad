# *** replicating a minimal toposort for the uop graph ***
from tingyrad import Tensor
# from tinygrad.uop.ops import UOp,Ops
t = Tensor([1,2,3])

"""dfs based toposort for the uop - replicating tinygrad behaviour"""
def toposort(u: UOp):
    stack = [(u, False)]
    cache = {}
    while stack:
        n,v = stack.pop()
        if node in cache: continue
        if not v:
            stack.append((n, True))
            for s in reversed(n.src): stack.append((s, False))
        else: cache[node] = None
    return cache

out = toposort(t.uop)
print(len(out))
