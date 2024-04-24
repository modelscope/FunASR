from funasr_onnx import CT_Transformer

# model_dir = "damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
model_dir = "damo/punc_ct-transformer_cn-en-common-vocab471067-large"
model = CT_Transformer(model_dir)

text_in = "跨境河流是养育沿岸人民的生命之源长期以来为帮助下游地区防灾减灾中方技术人员在上游地区极为恶劣的自然条件下克服巨大困难甚至冒着生命危险向印方提供汛期水文资料处理紧急事件中方重视印方在跨境河流问题上的关切愿意进一步完善双方联合工作机制凡是中方能做的我们都会去做而且会做得更好我请印度朋友们放心中国在上游的任何开发利用都会经过科学规划和论证兼顾上下游的利益"
result = model(text_in)
print(result[0])
