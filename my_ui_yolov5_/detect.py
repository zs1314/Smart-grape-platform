# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (MacOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

""" 导入一些库"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()  # __file__表示当前执行的预测文件的相对路径，FILE表示绝对路径（D:\yolov5-6.1\detect.py）
ROOT = FILE.parents[0]  # YOLOv5 root directory，获得项目的父目录——整个yolov5项目（D:\yolov5-6.1）
if str(ROOT) not in sys.path:  # sys.path表示模块查询路径的列表（判断这个yolov5项目是否在查询路径中）
    sys.path.append(str(ROOT))  # add ROOT to PATH——若不在，则添加进去（确保下面导包能成功）
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative，绝对路径转换为相对路径(转化为.)
# print(ROOT)

"""导入一些相对路径下的模块 """
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)  # 将待预测的东西的路径转为字符串
    # nosave默认位false,则not nosave为true,且source(传入的资源路径)的后缀是否为txt文件（一般若不想保存处理后的图片和视频，就将传入的nosave参数设为true）
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    # 判断传入资源（source）是否为文件地址——suffix表示后缀名（如.jpg）,[1:]表示从j开始 所以Path(source).suffix[1:]表示jpg、mp3等 IMG_FORMATS = [
    # 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp']  # include image suffixes VID_FORMATS = [
    # 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'wmv']  # include video suffixes
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    # 判断是否为网络流地址——startswith()判断地址开头是否是以下格式，lower()表示全部小写
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    # isnumeric()判断是否为数字——摄像头路径（比如0就是电脑的第一个摄像头）
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    # 判断网络流或地址是否都为true，若是则会下载那些网络图片、视频等
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    # project=ROOT / 'runs/detect'，ROOT就是. ，然后再拼接一个name(exp)——存放结果的最终文件名
    # increment_path表示一个增量，例如exp1、2、3
    # exist_ok表示是否新建一个文件夹（递增），还是一直在Path(project) / name中，默认为false，新建文件夹
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # save_txt——是否将结果预测以坐标的形式保存在txt文件中，默认为false，
    # mkdir——创建新文件
    # 这行就是表示如果设置save_txt为true，则将结果坐标以txt文件形式保存在save_dir / 'labels'中（runs/detect/exp3/labels）
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    # 自动选择设备，有GPU就是GPU
    device = select_device(device)
    # DetectMultiBackend()检测模型用的哪个框架（Pytorch等）
    # weights, device=device, dnn=dnn, data=data，分别表示传入的权重、设备、、传入的数据的文件路径（yaml类型）
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    # 得到模型的一些参数
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    # stride为32，check_img_size()判断传入的imgsz（图片尺寸）是否为32的倍数
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half，半精度，默认为false，可以有一个加速
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    # 是否为视频、网络流
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    #  是否为单张图片
    else:
        # 加载、初始化数据
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        # 每次输入一张图片
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference（推理部分）
    # 起到热身作用（一种训练技巧）
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    # 遍历数据加载模块
    # 返回的分别是：图片路径、resize后的图片（[3,640,480]）、原图、none、打印的字符串信息
    # 已一个一个的batch_szie为一轮，去循环。如果是图片，那batch为1，支循环一次
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        # 转换为pytorch支持tensor的格式 torch.size([3,640,480])
        im = torch.from_numpy(im).to(device)
        # 判断模型有无用到半精度
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        # 扩展一个batch维度——[1,3,640,480]
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference(预测)
        # 在模型预测过程中，把一些特征图保存下来，默认为false
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        # 开始预测，im——预处理完毕的图片，augment——增强conf的参数（默认为false）
        # 85——4个框的坐标信息，1个置信度信息，80个类别的概率
        # 18900——框的数目
        pred = model(im, augment=augment, visualize=visualize)  # torch.size([1,18900,85])
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS（非极大值抑制）——过滤
        # 传入参数——初始预测信息、置信度阈值、iou阈值、类别、、一张图最多检测出多少目标
        # 6——4个坐标（x,y,w,h，类别，conf），5——得到5个检测框，1——1个batch
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)  # 1,5,6
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # 图片的存储路径
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            # 获得原图的高和宽
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # 是否要将检测框单独裁剪下来
            imc = im0.copy() if save_crop else im0  # for save_crop
            # 画图工具
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            # det:torch.size([5,6]),5个框，6个相关信息
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    # 统计所有框的类别给n，方便后面答应出来
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        # 是否隐藏标签、置信度（图片中框上方）
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        # 画框
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        # 是否要将目标框截下来保存为图片（默认为false）
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            # 返回一个画好框的图片
            im0 = annotator.result()
            # 判断显示图片（默认为false）
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            # 保存结果
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
    return im0,s
    # Print results（打印结果）
    # dt——耗时时间
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        # 保存下来的提示信息
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

# 传入一些设定的参数
def parse_opt(img_path):
    """ default 为默认值"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/exp/weights/best.pt',
                        help='model path(s)')
    # 这个参数通过下载指定的模型（官方已经训练好的，包含全部权重）——default
    # http://admin:admin@192.168.2.108:8081
    parser.add_argument('--source', type=str, default=img_path, help='file/dir/URL/glob, 0 for webcam')
    # 这个就是预测的方式（图片、视频、摄像头都可以），通过调节default
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    # 置信度，当一个物体的置信度大于我们自己规定的值时，才用框框显示出目标(可以设置一下变量，然后在gui界面实时操控)
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    # iou——交并比，NMS非极大值抑制，当iou不超过（小于）我们设定的值，这个框就会抑制，直接变为0，若iou越大，那同一个物体的ground就越多
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # 设备

    """ 前面都是设置了默认参数，后面的只要用了，都会设置为true（如果要用非命令行，就从pycharm中设置一下参数）"""
    parser.add_argument('--view-img', action='store_true', help='show results')
    # 展示图片处理后的结果图片
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # 将结果预测以坐标的形式保存在txt文件中
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # 不保存结果
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    # 只检测某一个（或几个）指定的类别
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # 设置后NMS更加强大
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    # 这个也是提升结果的方式
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    # 把结果保存在哪个位置
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # 设置这个参数后，不会在旧文件后新建一个文件夹，而是把结果保存在旧文件中
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    # 对imagesize作额外判断——如果传入的参数只有1位（640），则会*2，变为640*640,若本来就有2个，则不变[640,640]
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    # 把所有的参数打印出来，opt就是存储了所有的参数
    print_args(FILE.stem, opt)
    # 将参数返回，传入给main()函数
    return opt


def main(opt):
    #  check_requirements检测requirements.txt文件中的包是否成功安装
    check_requirements(exclude=('tensorboard', 'thop'))
    # 将参数传入run()函数，**表示可以传入任意个参数
    return run(**vars(opt))


if __name__ == "__main__":
    img_path=r'D:\my_project_deeplearning\my_ui_yolov5_\data\images\bus.jpg'
    opt = parse_opt(img_path)  # 解析传入的参数
    main(opt)
