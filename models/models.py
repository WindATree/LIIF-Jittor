import copy
import jittor as jt

models = {}

# 构造模型,使用装饰器
# 比如定义一个类是 class edsr, 那么 register("edsr") 对应的字典就是 {"edsr" : <class '__main__.edsr'>}
def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator

def make(model_spec, args=None, load_sd=False):
    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        model_args = model_spec['args']
    
    # 创建模型实例
    model = models[model_spec['name']](**model_args)
    
    # 加载预训练权重
    if load_sd:
        state_dict = model_spec['sd']
        model.load_state_dict(state_dict)
    
    return model