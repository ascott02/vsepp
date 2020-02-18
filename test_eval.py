from vocab import Vocabulary
import evaluation

model_path = "./runs/coco_vse++_resnet_restval/model_best.pth.tar" # pretrained image model
data_path="./data/"
split = "test"

evaluation.evalrank(model_path, data_path=data_path, split=split)

