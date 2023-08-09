#komut satırlalarını kolyaca yazabilmek
import argparse
import time
#dosya yolları ve dosya işlemleri için kullanılır
from pathlib import Path
#nesne takbibi 
import cv2
#yolov7 pytorch kullanılarak oluşturulduğundan yolov7 kullanarak nesne algılama izlemsini gerçekleştirmek için
#pytorch modüdlünü geliştirmemiz gerek
import torch
#pytorch un desteklediği çeşitli backendleri kontrol etmek için kullanılan mödülün içe aktarılması gerekir
import torch.backends.cudnn as cudnn
#rastgele sayi üretmek 
from numpy import random
#nesne tespiti ve takip için kullanılan modelleri yüklemek için
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
#tespit edilen nesnenin kutu çizimi
from utils.plots import plot_one_box
# torch ile ilgili yardımcı kütüphaneler eklenir
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
#deepSort algoritması 
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
#oluşturulan listeleri diziye dönüştürmek 
from collections import deque
import numpy as np
import math

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

#reklam iconlarının isismlerini tutan sözlük 
data_deque = {}
#tespit edilen neslerin sayısını tuatn sözlük
object_counter={}

#bu fonksiyon yolov7'den alınan sınırlayıcı kutuları DeepSort ile uyumlu formata dönüştürmek için kullanılan fonksiyonu içerir
def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h
# x, y koordinatlarını sağ alt ve sol üst köşelerinin koordinatlarını temsil eder ve oluşan değerleri listelere ekler
def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs
#eğitilen videodaki nesnelerin etiketleri
def compute_color_for_labels(label):
    if label == 0: #kanald iconu
        color = (0, 102, 204)
    elif label == 1: 
        color = (0, 0, 0) #reklamlar iconu
    elif label == 2:  
        color = (0, 0, 0) # reklam yazisi
    elif label == 3:  
        color = (255, 128, 0) # show iconu
    elif label == 4:  
        color = (0, 0, 0) # reklam iconu
    elif label == 5:  
        color = (102, 0, 102) #star iconu
    elif label == 6:  
        color = (0, 0, 0) # reklam iconu
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)
#görüntü üzerinde yuvarlak kenarlı bir dikdörtgen çizmek ve bu dikdörtgenin içerisinde icon isimlerini gösterilmesini sağlar.
def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    # sol üst taraf
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # sağ üst taraf
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # sol alt taraf
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # sağ alt taraf
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    
    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)
    
    return img
#bu fonksiyon tespit edilen nesne etrafında dikdörtgenleri çizer ve dikdörtgenlerin içerisine etiket ekler
def UI_box(x, img, color=None, label=None, line_thickness=None):
   
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    
    if label:
        tf = max(tl - 1, 1)  
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

#bu fonksiyonda nesnelerin etrafında dikdörtgenleri çizerek her nesneye benzersiz kimlik atayarak 
#iconları daha rahat tespit etmemizde işe yarar
def draw_boxes(img, bbox, names, object_id, identities=None, offset=(0, 0)):
   #kare kare algılama yaptığımız için mevcut karenin yük yükseliği ve genilşiği kontrol edilir
    height, width, _ = img.shape 
    #her iconun biribirine benzemeyen idsini atayabilmek için sözlük oluşturulur eğer nesne daha önce görülmemişse yeni bir deque oluşturulur
    for key in list(data_deque):
      if key not in identities:
        data_deque.pop(key)

    #kutucunun korrdinatları offset sayesinde güncellenir.
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        #kutunun merkezinin bulunması
        center = (int((x2+x1)/ 2), int((y2+y2)/2))

        #iconlara id oluşturma
        id = int(identities[i]) if identities is not None else 0
        #id atadındaktan sonra maksimum uzunluğu 64 olan yeni nesne için yeni arabellek oluştrulur.
        #eğer nesne daha önce görülmemişse sözlüğüne ywni bir kuyruk eklenir
        if id not in data_deque:  
        # kuruğun en fazla boyutu
          data_deque[id] = deque(maxlen= opt.trailslen)
        #coco.names ile yazılan icon isimleri idler kullanarak icon isimlerini buluuruz
        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        label ='%s' % (obj_name)
        
        data_deque[id].appendleft(center)
        
        #nesne sayacını güncelleştirme ve görselleştirme bölümü
        if len(data_deque[id]) >= 2:
                if obj_name not in object_counter:
                    object_counter[obj_name] = 1
                else:
                    #icon ismi sözlüğe daha önceden eklenmişse değerini 1 arttırıyoruz
                    object_counter[obj_name] += 1

        UI_box(box, img, label=label, color=color, line_thickness=2)
        #nesnenin takip ettiği izleri temsil eder
        for i in range(1, len(data_deque[id])):
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            thickness = int(np.sqrt(opt.trailslen / float(i + i)) * 1.5)
            #takip çizigisi görüntüye eklenmesi
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)
    #reklamlar ve diziye devam edildiğinde ekranın orta kısmında yazının gösterilmesi
    for idx, (key, value) in enumerate(object_counter.items()):
          cnt_str = str(key) 
          if cnt_str in obj_name:
            cv2.line(img, (height - 190 ,25+ (idx*40)), (height,25 + (idx*40)), [85,45,255], 30)
            cv2.putText(img, cnt_str, (height - 190, 35 + (idx*40)), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)     
    return img

# dosyaları açıp okuma names metin dosyası coco.names
def load_classes(path):
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names)) 
#nesne tespiti, çıktıları görüntüleme sonuçları metin dosyasına kaydeder
def detect(save_img=False):
    names, source, weights, view_img, save_txt, imgsz, trace = opt.names, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace 
    save_img = not opt.nosave and not source.endswith('.txt')  
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    #Kaydedilecek dosya ve klasörlerin yolunu temsil eder
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True) 
    #DeepSort algoritmasının başaltılması
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    #modelin yüklenmesi ve yapılandırılması
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu' 

    model = attempt_load(weights, map_location=device) 
    stride = int(model.stride.max()) 
    imgsz = check_img_size(imgsz, s=stride) 
    #İzlenilmiş modelin daha hızlı çalışmasını sağlar
    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half() 

    #eğitilmiş sınıflandırıcıyı yükler
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    #veri yükleyicinin yapılandırılması ve çıktının çalıştrılması
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    #isimlerin ve renklerin alınması
    names = load_classes(names)

    #modelin bir kez çalıştırılmasını sağlar. Bellek ayarlamaları
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float() 
        img /= 255.0 
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        #gpu kullanıyorsa ve görüntü boyutu değişmemişse modelin çalışmasını sağlamak için tekrar çalıştırır
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        
        t1 = time_synchronized()
        #gpu bellek sızınıtısını önlemek için gereklidir
        with torch.no_grad():   
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()
        #tahminler üzerinde en fazla nesne bastırma işlemini uygular 
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        #nesne bastırma işlemi tamamlandı
        t3 = time_synchronized()
        #ikinci aşama sınıflandırıcı, tahminler üzerinde sınıflandırma yapılır sınıflandırma sonuçlarını güncellenir
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        #görüntü üzerinde dolaşır
        for i, det in enumerate(pred):
            if webcam:  #webcamden gelen görüntüler varsa onların işlendiği görünütleri kontrol eder
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
       
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                #sonuçları yazdırma
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  
                    s += '%g %ss, ' % (n, names[int(c)]) 
                #koordinatları ve boyutları içeren liste
                xywh_bboxs = []
                # güven değerlerini içeren liste
                confs = []
                # nesne id'leri
                oids = []
                #Sonuçları yazırma
                #tespit edilen iconlar üzerinde dolaşır, elde edilen koordinat ve boyutlar listeye eklenir.
                for *xyxy, conf, cls in reversed(det):
                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                    xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                    xywh_bboxs.append(xywh_obj)
                    confs.append([conf.item()])
                    oids.append(int(cls))
                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)
                #deepsort sayesinde tespit ettiğimiz iconları alıp kimlikleri ekliyoruz
                outputs = deepsort.update(xywhs, confss, oids, im0)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    object_id = outputs[:, -1]
                    #sınırlayıcı kutucukları çizer ve etiketleri ekler ve idlerini gösterir
                    draw_boxes(im0, bbox_xyxy, names, object_id ,identities)
            #video da tespit ediken süreleri obje isimlerini yazdırır
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            #Sonuçları görüntülerde canlı olarak yayınlayarak sonuçları videolar üzerinde kaydetmemizi sağlar
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  
            #sonuçları video üzerinde kaydetne
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  
                    if vid_path != save_path: 
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  
                        if vid_cap:  
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''

    #işlem tamamlanma süresi
    print(f'Done. ({time.time() - t0:.3f}s)')

# yolov7 modeli
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7_training.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source') 
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.cfg path')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--trailslen', type=int, default=64, help='trails size (new parameter)')
    opt = parser.parse_args()
    print(opt)
  
    #modelin eğitim modunda olmadan çıkarım yapmasını sağlar 
    with torch.no_grad():
        if opt.update: 
            #güncellenecek modelin dosya yolunu belirtir
            for opt.weights in ['yolov7_training.pt']:
                detect()
                strip_optimizer(opt.weights)
        #nesne tespitini gerçekleştirir        
        else:
            detect()
