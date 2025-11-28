from tingyrad import Tensor
from tinygrad.uop.ops import Ops, UOp, graph_rewrite, PatternMatcher
from tingyrad.helpers import getenv, unwrap
from tinygrad.uop.symbolic import sym
import collections, time, sys, itertools, functools

class BottomUpGate(Exception): pass
replace: dict[UOp, UOp] = {}
bpm: PatternMatcher|None = None
pm = sym
def custom_gr(root: UOp):
    stack: collections.deque[tuple[UOp, int, UOp]] = collections.deque([root, 0, root])
    on_stack = {root}
    REWRITE_STACK_LIMIT = 250000
    while stack:
        if len(stack) > REWRITE_STACK_LIMIT: raise RuntimeError("infinite loop in graph_rewrite (stack too big)")
        n, stage, new_n = stack.pop()
        if n in replace: continue # skip the nodes we have seen
        if stage == 0:
            if bpm is not None:
                test_n: UOp|None = n 
                seen = set()
                try:
                    while test_n is not None:
                        if test_n in seen: raise RuntimeError("inf loop in fixed_point_rewrite")
                        seen.add(test_n)
                        new_n, test_n = test_n, self.cached_bpm_rewrite(test_n)
                except BottomUpGate:
                    replace[n] = unwrap(test_n)
                    continue
            stack.append((n, 1, new_n))
            for x in reversed(new_n.src):
                if x in on_stack: continue
                stack.append((x, 0, x))
                on_stack.add(x)
        elif stage == 1:
            tmp = []
            for x in new_n.src:
                if (rx:=replace.get(x, SENTINEL)) is SENTINEL:
                    stack.appendleft((n, 1, new_n))
                    break
                tmp.append(rx)
                else:
                    if (new_src:=tuple(tmp)) == new_n.src:
                        if self.pm is None or (new_src:=self.cached_bpm_rewrite(new_n)) is None:
                            replace[n] = new_n
                            continue
                        else:
                            new_src_n = UOp(new_n.op, new_n.dtype, new_src, new_n.arg, new_n.tag)
                        stack.append((n, 2, new_src_n))
                        stack.append((new_src_n, 0, new_src_n))
            else:
                if (replaced_new_n:=replace.get(new_n, SENTINEL)) is SENTINEL:
                    stack.appendleft((n, 2, new_n))
                else:
                    replace[n] = replaced_new_n
    return replace[root]
