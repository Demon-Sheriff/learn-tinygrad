# *** performing a DFS on the generated uop graph, just for fun ***
from tinygrad import Tensor
# this is ofcourse without calling realize, numpy, or item or anything that triggers computation.

t = Tensor([1,2,]) # buffer creation
t1 = t + 1
# r = t + 1 # broadcast
# t = t.exp() # unary op 

# till now no computation has been triggered and the graph is actually a linkedlist as of now.
# ideally we should be able to perform a dfs traversal on this using the source property.

# t = t.realize()

# so realize triggers kernelize first which also forces graph_rewirte hence in the viz tool after realize we see a rather simplified graph.
# so ideally learning how the graph_rewrite engine works should be the ideal next step.

l = [x.uop for x in t] # i still can't figure out how each element of the tensor beomces a uop in this case
# also each scalar within the tensor is kinda wrapped into an Ops.Vectorize, i'm guessing this is because the tensor has to be vectorized for operations.

# print(t.uop.op)
# cnt = 0
def dfs(root_uop):
    # cnt += 1
    print(f"ops for uop : {root_uop.op}")
    print(f"arg for uop : {root_uop.arg if ((r:=root_uop.arg) != () or root_uop.arg is not None) else 'empty arg'}")
    if root_uop.src is None or root_uop.src == ():
        print("reached the leaf\n")
        return
    for s in root_uop.src:
        dfs(s)

dfs(t1.uop)
print(t1.uop)

# for each element in the tensor
for x in t1:
    dfs(x.uop)
    print()

# print(t.numpy())
for k,v in t1.uop.reverse_toposort(t1.uop.get_consumer_map()).items():
    # print(v)
    print(k.op)
    print(k)

# for k,v in t.uop.toposort().items():
#     print(k.op)
# print(t.uop.toposort())

print(t1.uop.src[1].op)