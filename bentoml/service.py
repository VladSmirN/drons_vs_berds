import bentoml
from bentoml.io import Image
from bentoml.io import PandasDataFrame
from ultralytics import YOLO
import pandas as pd 

class Yolov8Runnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        import torch
  
        # self.model = torch.hub.load("ultralytics/yolov5:v6.2", "yolov5s")
        self.model = YOLO('./models/yolo8n_best.pt')
        if torch.cuda.is_available():
            self.model.cuda()
        else:
            self.model.cpu()

        # Config inference settings
        self.inference_size = 320

 

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def inference(self, input_img):
 
        result = self.model(input_img )[0]
        return pd.DataFrame(result.boxes.xyxy.cpu().numpy(),columns=['x1','y1','x2','y2'])

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def render(self, input_img):
 
        return self.model(input_img )[0].plot()[:,:,::-1]


yolo_v8_runner = bentoml.Runner(Yolov8Runnable, max_batch_size=30)

svc = bentoml.Service("drons_detection", runners=[yolo_v8_runner])


@svc.api(input=Image(), output=PandasDataFrame())
async def invocation(input_img):
    batch_ret = await yolo_v8_runner.inference.async_run([input_img])
    return batch_ret


@svc.api(input=Image(), output=Image())
async def render(input_img):
    batch_ret = await yolo_v8_runner.render.async_run([input_img])
    return batch_ret