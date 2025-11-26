from tinygrad import Tensor
from tinygrad.uop.ops import GroupOp, UPat, Ops, Invalid

a = Tensor([1,2,3])
b = Tensor([3,4,5])
c = a@b
# d = c.relu()
# c.realize()
print(c.numpy())

invalid_pat = UPat(Ops.CONST, arg=Invalid, name="i")
invalid_gate = UPat.var("cond").where(UPat.var("x"), invalid_pat)

k = [*((invalid_gate.alu(op, UPat.var("y")).named("alu"), lambda cond,x,y,alu,i: cond.where(x.alu(alu.op,y), i))
    for op in GroupOp.Binary-GroupOp.Comparison)]
print(k)