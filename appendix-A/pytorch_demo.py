import torch

print(torch.__version__)
print(torch.mps.is_available())

tensor2d = torch.tensor([[1,2,3],[3,2,1]])
print(tensor2d.shape)

print(tensor2d.reshape(3,2))
print(tensor2d.shape)

print("tensordemo")
tensordemo = torch.arange(12)
print(tensordemo)
reshapeTensorDemo = tensordemo.reshape(3,4)
print(reshapeTensorDemo)
print(torch.zeros((2, 3, 4)))

randomTensorDemo = torch.randn(3,4)
print("randomTensorDemo",randomTensorDemo)

specialTensorDemo = torch.tensor([2,3])
print("specialTensorDemo", specialTensorDemo)


tesorx = torch.tensor([1,2,3,4])
tesory = torch.tensor([3,4,5,6])
tesorRes = tesory * tesorx
print("tesnsrmultiPy Result ", tesorRes);

beforeReshape = torch.arange(6)
print("beforeReshape" ,beforeReshape)
afterReshape = beforeReshape.reshape((3, 2))
print("afterReshape",afterReshape)

a = torch.arange(3).reshape(3, 1)
print("a " ,a)
b = torch.arange(2).reshape(1, 2)
print("b ", b)
combine = a + b
print("a + b  combine",combine)

print(combine[-1])
print(combine[1:2])

combine[1,1] = 12111
print(combine)


tensor1d = torch.tensor([1,23])
print("tensor1d type",tensor1d.dtype)
tensor1dv = tensor1d.to(torch.float32)
print("tensor1dv", tensor1dv.dtype)

tensor1dv1 = torch.arange(12).reshape(3,4)
print("tensor1dv1 ", tensor1dv1.shape)


# print("tensor1dv1 @ ", tensor1dv1 @ tensor1dv.T)
