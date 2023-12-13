import pathlib
import shutil

from ultralytics import YOLO
from utils import get_base_arg_parser


def main(args):
    model = YOLO(args.model_path, task='detect')

    model.export(format='onnx', dynamic=True, simplify=True)

    onnx_model_path = pathlib.Path(args.model_path).with_suffix('.onnx')

    # dir used by the savant pipeline
    savant_model_cache_dirpath = pathlib.Path('/cache/models/yolov8_pipeline/yolov8m')
    savant_model_cache_dirpath.mkdir(parents=True, exist_ok=True)

    # move onnx model to savant model cache dir
    (savant_model_cache_dirpath / onnx_model_path.name).unlink(missing_ok=True)
    shutil.move(str(onnx_model_path), savant_model_cache_dirpath)

    # write labelfile
    labels_file_path = savant_model_cache_dirpath / 'labels.txt'
    with open(labels_file_path, 'w', encoding='utf8') as filestream:
        for _, label in sorted(model.names.items()):
            filestream.write(f'{label}\n')


if __name__ == '__main__':
    parser = get_base_arg_parser()
    main(parser.parse_args())
