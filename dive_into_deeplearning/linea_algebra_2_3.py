import torch


A = torch.arange(12).reshape(3, 4)
B = torch.arange(20).reshape(4, 5)


print(A)
print(B)
print(torch.mm(A,B))

print(B.T )