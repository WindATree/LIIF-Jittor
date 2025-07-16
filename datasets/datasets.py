import copy


datasets = {}

# 构造数据集,使用装饰器
# 比如定义一个类是 class MNISTDataset, 那么 register("mnist") 对应的字典就是 {"mnist" : <class '__main__.MNISTDataset'>}
def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator

# make 函数根据 yaml 配置文件来实例化数据集对象
def make(dataset_spec, args=None):
    if args is not None:
        dataset_args = copy.deepcopy(dataset_spec['args'])
        dataset_args.update(args)
    else:
        dataset_args = dataset_spec['args']
    dataset = datasets[dataset_spec['name']](**dataset_args)
    return dataset