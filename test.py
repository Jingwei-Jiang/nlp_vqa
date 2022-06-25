from dataset import *

loader=get_loader(train = True).create_dict_iterator()
print("Loader Created")

print("iter:0")
test = loader._get_next()
print(test['q'])
print(test['a'])
print(test['img'].shape)

print("iter:1")
test = loader._get_next()
print(test['q'])
print(test['a'])
print(test['img'].shape)
q_array = test['q'].asnumpy()
print(q_array[0])