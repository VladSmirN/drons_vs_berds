#source - https://github.com/yihong1120/Construction-Hazard-Detection/blob/main/examples/YOLOv8-Evaluation/evaluate_sahi_yolov8.py

import argparse
import numpy as np
import os
from typing import List, Dict
from tqdm import tqdm

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.coco import Coco
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import json

class COCOEvaluator:
    """
    Evaluates object detection models using COCO metrics.
    """

    def __init__(self, model_path: str, coco_json: str, image_dir: str, model_type:str, confidence_threshold: float = 0.5,
                 slice_height: int = 640, slice_width: int = 640, overlap_height_ratio: float = 0.2,
                 overlap_width_ratio: float = 0.2, save_path='./eval.json', config_path=''):
        """
        Initialises the evaluator with model and dataset parameters.

        Args:
            model_path (str): Path to the trained model file.
            coco_json (str): Path to the COCO format annotations JSON file.
            image_dir (str): Directory containing the evaluation image set.
            confidence_threshold (float, optional): Confidence threshold for predictions. Defaults to 0.3.
            slice_height (int, optional): Height of the slices for prediction. Defaults to 640.
            slice_width (int, optional): Width of the slices for prediction. Defaults to 640.
            overlap_height_ratio (float, optional): Overlap ratio between slices in height. Defaults to 0.2.
            overlap_width_ratio (float, optional): Overlap ratio between slices in width. Defaults to 0.2.
        """

        if model_type == 'yolov8':
            self.model = AutoDetectionModel.from_pretrained(
                model_type=model_type,
                model_path=model_path,
                confidence_threshold=confidence_threshold,
  
            )
        if model_type == 'detectron2':
            self.model = AutoDetectionModel.from_pretrained(
                model_type=model_type,
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                config_path=config_path
            )
  
        self.coco_json = coco_json
        self.image_dir = image_dir
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.save_path=save_path
        self.model_type = model_type

    def evaluate(self) -> Dict[str, float]:
        """
        Performs the evaluation of the model against the given dataset and computes COCO metrics.

        Returns:
            Dict[str, float]: A dictionary containing computed metrics.
        """
        coco = Coco.from_coco_dict_or_path(self.coco_json)
        pycoco = COCO(self.coco_json)
        predictions = []
        category_to_id = {category.name: category.id for category in coco.categories}

        for image_info in tqdm(coco.images, desc='Progress of predictions'):
            image_path = os.path.join(self.image_dir, image_info.file_name)
            prediction_result = get_sliced_prediction(
                image_path, self.model, slice_height=self.slice_height, slice_width=self.slice_width,
                overlap_height_ratio=self.overlap_height_ratio, overlap_width_ratio=self.overlap_width_ratio, verbose=0
            )
            for pred in prediction_result.object_prediction_list:

                if self.model_type == 'yolov8':
                    category_id = category_to_id[pred.category.name]

                if self.model_type == 'detectron2':
                    category_id = 1    

                predictions.append({
                    "image_id": image_info.id,
                    "category_id": category_id,
                    "bbox": [
                        float(pred.bbox.minx), float(pred.bbox.miny),
                        float(pred.bbox.maxx - pred.bbox.minx), float(pred.bbox.maxy - pred.bbox.miny)
                    ],
                    "score": float(pred.score.value),
                })

        with open(self.save_path, "w") as file:
            file.write(json.dumps(predictions)) 

        pycoco_pred = pycoco.loadRes(predictions)
        coco_eval = COCOeval(pycoco, pycoco_pred, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        metrics = {
            "Average Precision": np.mean(coco_eval.eval['precision'][:, :, :, 0, -1], axis=(0, 1, 2)),
            "Average Recall": np.mean(coco_eval.eval['recall'][:, :, 0, -1], axis=(0, 1)),
            "mAP at IoU=50": np.mean(coco_eval.eval['precision'][0, :, :, 0, 2]),
            "mAP at IoU=50-95": np.mean(coco_eval.eval['precision'][0, :, :, 0, :])
        }
        return metrics

def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluates model using COCO metrics.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--coco_json", type=str, required=True, help="Path to the COCO format annotations JSON file.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing the evaluation image set.")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--slice_size", type=int, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--config_path", type=str )
   
    # parser.add_argument("--overlap_ratio", type=int, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    evaluator = COCOEvaluator(
        model_path=args.model_path,
        coco_json=args.coco_json,
        image_dir=args.image_dir,
        save_path=args.save_path,
        slice_height=args.slice_size,
        slice_width=args.slice_size,
        model_type=args.model_type,
        config_path=args.config_path
    )
    metrics = evaluator.evaluate()
    print("Evaluation metrics:", metrics)

"""example usage
python evaluate_model.py --model_path "../../models/best_yolov8n.pt" --coco_json "/Users/YiHung/Documents/Side_Projects/Construction-Hazard-Detection/examples/YOLOv8-Evaluation/dataset/coco_annotations.json" --image_dir "/Users/YiHung/Documents/Side_Projects/Construction-Hazard-Detection/examples/YOLOv8-Evaluation/dataset/valid/images"
"""