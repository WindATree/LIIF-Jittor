import copy
import jittor as jt

models = {}

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
    
    # 加载预训练权重（适配 Jittor 的 load_state_dict 实现）
    if load_sd:
        # 确保权重字典中的键与模型参数匹配
        state_dict = model_spec['sd']
        model.load_state_dict(state_dict)
    
    return model