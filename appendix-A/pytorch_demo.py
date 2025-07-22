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
