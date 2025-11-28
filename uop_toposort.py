# *** replicating a minimal toposort for the uop graph ***
from tinygrad import Tensor
from tinygrad.uop.ops import UOp,Ops
t = Tensor([1,2])
t1 = t + 1
# t2 = t1 @ Tensor([3,4,5])

"""dfs based toposort for the uop - replicating tinygrad behaviour"""
def toposort(u: UOp):
    # the general idea is to resolve a node's children first
    # if the it's children have been resolved then this can be marked true
    stack = [(u, False)]
    cache = {} # to store the topological order
    while stack:
        node, vis = stack.pop()
        if node in cache: continue # if the node is already been resolved progress to the next node
        if vis: cache[node] = None # second time i'm seeing this node, all children have been resolved, place it.
        else:
            stack.append((node, True)) # first time seeing this, schedule a second pass
            # resolve the children
            for s in reversed(node.src): stack.append((s, False))

    return cache

def consumer_map_from_toposort(lst):
  ret: dict[UOp, dict[UOp, None]] = {}
  for u in lst:
    ret[u] = {}
    for s in u.src: ret[s][u] = None
  return ret

def get_consumer_map(u):
    return consumer_map_from_toposort(u.toposort())

out = toposort(t1.uop)
# print(len(out))
# print(out)
for k,v in out.items():
    print(k.op)

# print(t1.uop)
print()
for k,v in get_consumer_map(t1.uop).items():
    print(v)

print()
print(t1.uop)
print(t1.uop.src[0].src)
