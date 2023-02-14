# 搭建自定义任务
FunASR类似ESPNet，以`Task`为通用接口，从而实现模型的训练和推理。下面我们将以paraformer模型为例，介绍如何定义一个新的`Task`。

`Task`是一个类，其需要继承`AbsTask`，其对应的代码见`funasr/tasks/abs_task.py`，