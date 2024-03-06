import os
from pprint import pprint
from paddlenlp import Taskflow
#schema = ["病情诊断", "治疗方案", "病因分析", "指标解读", "就医建议", "疾病表述", "后果表述", "注意事项", "功效作用", "医疗费用", "其他"]
#my_cls = Taskflow("zero_shot_text_classification", model="utc-base", schema=schema, task_path='./checkpoint/model_best/plm', precision="fp16")
schema = ["无线电实时监测","无线电查询正常监测站数量","无线电查询异常监测站数量","无线电查询非法信号监测站数量","无线电查询非法信号数量","无线电查询信号数量","无线电查询合法信号数量","无线电查询航空干扰数量"]
cwd = os.getcwd()
path = os.path.join(cwd, "checkpoint/model_best/plm")
task_path="../../checkpoint/model_best/plm"
my_cls = Taskflow("zero_shot_text_classification", model="utc-base", schema=schema,  precision="fp16",task_path=task_path)
# realtime
result=my_cls("让黄山站做测向任务带宽100k")
assert(len(result)==1)
assert(len(result[0]["predictions"])>0)
label=result[0]["predictions"][0]["label"]
assert(label=="无线电实时监测")


result=my_cls("齐齐哈尔站单频测量频率100")
assert(len(result)==1)
assert(len(result[0]["predictions"])>0)
label=result[0]["predictions"][0]["label"]
assert(label=="无线电实时监测")




result=my_cls("正常监测站有多少")
pprint(result)
assert(len(result)==1)
assert(len(result[0]["predictions"])>0)
label=result[0]["predictions"][0]["label"]
assert(label=="无线电查询正常监测站数量")

result=my_cls("昨天正常监测站有多少")
assert(len(result[0]["predictions"])>0)
label=result[0]["predictions"][0]["label"]
assert(label=="无线电查询正常监测站数量")

result=my_cls("2023年6月10号正常监测站有多少")
assert(len(result[0]["predictions"])>0)
label=result[0]["predictions"][0]["label"]
assert(label=="无线电查询正常监测站数量")

result=my_cls("现在有多少个监测站正常")
assert(len(result[0]["predictions"])>0)
label=result[0]["predictions"][0]["label"]
assert(label=="无线电查询正常监测站数量")

result=my_cls("2023年1月1日至2024年5月20日有多少监测站在正常工作")
assert(len(result[0]["predictions"])>0)
label=result[0]["predictions"][0]["label"]
assert(label=="无线电查询正常监测站数量")

result=my_cls("2023年1月1日至2024年5月20日有多少监测站无法工作")
assert(len(result[0]["predictions"])>0)
label=result[0]["predictions"][0]["label"]
assert(label=="无线电查询异常监测站数量")

result=my_cls("有多少监测站坏了")
assert(len(result[0]["predictions"])>0)
label=result[0]["predictions"][0]["label"]
assert(label=="无线电查询异常监测站数量")
