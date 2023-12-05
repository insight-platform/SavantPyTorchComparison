from ultralytics import YOLO
from utils import get_base_arg_parser


def main(args):
    model = YOLO(args.model_path, task='detect')
    model.export(format='onnx', dynamic=True, simplify=True)


if __name__ == '__main__':
    parser = get_base_arg_parser()
    main(parser.parse_args())
