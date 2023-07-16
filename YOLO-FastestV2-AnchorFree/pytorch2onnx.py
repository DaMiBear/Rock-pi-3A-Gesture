import argparse

import torch
import models.detector
import build_utils.utils
from build_utils.parse_config import parse_data_cfg

if __name__ == '__main__':
    #指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/mydata.data', 
                        help='Specify training profile *.data')
    parser.add_argument('--weights', type=str, default='weights/c430v2-07401-olddataset init.pt', 
                        help='The path of the .pth model to be transformed')

    parser.add_argument('--output', type=str, default='./YOLO-FastestV2-AnchorFree.onnx', 
                        help='The path where the onnx model is saved')

    opt = parser.parse_args()
    data_dict = parse_data_cfg(opt.data)
    data_dict["classes"] = int(data_dict["classes"])
    data_dict["anchor_num"] = int(data_dict["anchor_num"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.detector.Detector(data_dict["classes"], data_dict["anchor_num"], init_backbone=False, use_focal_loss=False, export_onnx=True).to(device)
    model.load_state_dict(torch.load(opt.weights, map_location=device)["model"])
    #sets the module in eval node
    model.eval()

    test_data = torch.rand(1, 3, 352, 352).to(device)
    torch.onnx.export(model,                    #model being run
                      test_data,                 # model input (or a tuple for multiple inputs)
                      opt.output,               # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=11,          # the ONNX version to export the model to
                      do_constant_folding=True)  # whether to execute constant folding for optimization

    

