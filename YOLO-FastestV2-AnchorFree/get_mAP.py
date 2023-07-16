import yaml
import torch
from build_utils.datasets_coco import LoadImagesAndLabels_COCO
from build_utils.parse_config import parse_data_cfg
from train_utils import train_eval_utils
from models.detector import Detector
from train_utils.coco_utils import get_coco_api_from_dataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = "data/mydata.data"
    weights_path = "weights/best.pt"   # 训练好的权重文件
    imgsz_test = 352  # 32整数倍
    batch_size = 32
    hyp = "cfg/hyp.yaml"
    with open(hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)
    data_dict = parse_data_cfg(data)
    data_dict["classes"] = int(data_dict["classes"])
    data_dict["anchor_num"] = int(data_dict["anchor_num"])
    test_dataset = LoadImagesAndLabels_COCO(path=data_dict["dataset_dir"],
                                            images_dir=data_dict["val_images_dir"],
                                            json_file=data_dict["val_json_file"],
                                            img_size=imgsz_test,
                                            batch_size=batch_size, augment=False, hyp=hyp, 
                                            rect=True,
                                            cache_images=False,
                                            single_cls=False)
    test_datasetloader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=batch_size,
                                                    num_workers=8,
                                                    pin_memory=True,
                                                    collate_fn=test_dataset.collate_fn)
    
    coco = get_coco_api_from_dataset(test_dataset)
    model = Detector(data_dict["classes"], data_dict["anchor_num"], init_backbone=False).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device)["model"])
    result_info = train_eval_utils.evaluate(model, test_datasetloader,
                                              coco=coco, device=device)

if __name__ == "__main__":
    main()