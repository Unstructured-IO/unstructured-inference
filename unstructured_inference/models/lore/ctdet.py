from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import cv2
import numpy as np
import torch

from unstructured_inference.models.lore.classifier import Processor
from unstructured_inference.models.lore.debugger import Debugger
from unstructured_inference.models.lore.decode import corner_decode, ctdet_4ps_decode
from unstructured_inference.models.lore.image import get_affine_transform_upper_left
from unstructured_inference.models.lore.model import create_model, load_model
from unstructured_inference.models.lore.post_process import ctdet_4ps_post_process_upper_left, ctdet_corner_post_process


class BaseDetector(object):
  def __init__(self, opt):
    # if opt.gpus[0] >= 0:
    #   opt.device = torch.device('cuda')
    # else:
    opt.device = torch.device('cpu')

    self.model = create_model(opt.arch, opt.heads, opt.head_conv)
    self.model = load_model(self.model, opt.load_model)
    self.model = self.model.to(opt.device)
    self.model.eval()

    self.processor = Processor(opt)
    self.processor = load_model(self.processor, opt.load_processor)
    # self.processor

    self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
    self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
    self.max_per_image = opt.K
    self.num_classes = opt.num_classes
    self.scales = opt.test_scales
    self.opt = opt
    self.pause = True

  def pre_process(self, image, scale, meta=None):
    height, width = image.shape[0:2]
    new_height = int(height * scale)
    new_width = int(width * scale)
    if self.opt.fix_res:
      inp_height, inp_width = self.opt.input_h, self.opt.input_w
      c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
      s = max(height, width) * 1.0
    else:
      inp_height = (new_height | self.opt.pad)  # + 1
      inp_width = (new_width | self.opt.pad)  # + 1
      c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
      s = np.array([inp_width, inp_height], dtype=np.float32)
    if self.opt.upper_left:
      c = np.array([0, 0], dtype=np.float32)
      s = max(height, width) * 1.0
      trans_input = get_affine_transform_upper_left(c, s, 0, [inp_width, inp_height])
    else:
      trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv2.resize(image, (new_width, new_height))
    inp_image = cv2.warpAffine(
      resized_image, trans_input, (inp_width, inp_height),
      flags=cv2.INTER_LINEAR)

    inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    if self.opt.flip_test:
      images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
    images = torch.from_numpy(images)
    meta = {'c': c, 's': s,
            'input_height': inp_height,
            'input_width': inp_width,
            'out_height': inp_height // self.opt.down_ratio,
            'out_width': inp_width // self.opt.down_ratio}
    return images, meta

  def save_img_txt(self, img):
    shape = list(img.shape)
    f1 = open('/home/rujiao.lrj/CenterNet_cell_Coord/src/img.txt', 'w')
    for i in range(shape[0]):
      for j in range(shape[1]):
        for k in range(shape[2]):
          data = img[i][j][k].item()
          f1.write(str(data) + '\n')
    f1.close()

  def Duplicate_removal(self, results, corners):
    bbox = []
    for j in range(len(results)):
      box = results[j]
      if box[-1] > self.opt.scores_thresh:
        for i in range(8):
          if box[i] < 0:
            box[i] = 0
          if box[i] > 1024:
            box[i] = 1024

        def dist(p1, p2):
          return ((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])) ** 0.5

        p1, p2, p3, p4 = [box[0], box[1]], [box[2], box[3]], [box[4], box[5]], [box[6], box[7]]
        if dist(p1, p2) > 3 and dist(p2, p3) > 3 and dist(p3, p4) > 3 and dist(p4, p1) > 3:
          bbox.append(box)
        else:
          continue

    corner = []
    for i in range(len(corners)):
      if corners[i][-1] > self.opt.vis_thresh_corner:
        corner.append(corners[i])
    return np.array(bbox), np.array(corner)

  def filter(self, image_name, results, logi, ps):
    # this function select boxes
    batch_size, feat_dim = logi.shape[0], logi.shape[2]
    num_valid = sum(results[1][:, 8] >= self.opt.vis_thresh)

    # if num_valid <= 900 : #opt.max_objs
    slct_logi = np.zeros((batch_size, num_valid, feat_dim), dtype=np.float32)
    slct_dets = np.zeros((batch_size, num_valid, 8), dtype=np.int32)
    for i in range(batch_size):
      for j in range(num_valid):
        slct_logi[i, j, :] = logi[i, j, :].cpu()
        slct_dets[i, j, :] = ps[i, j, :].cpu()
    # else:
    # print('Error: Number of Detected Boxes Exceed the Model Defaults.')
    # quit()

    return torch.Tensor(slct_logi), torch.Tensor(slct_dets)

  def process_logi(self, logi):
    logi_floor = logi.floor()
    dev = logi - logi_floor
    logi = torch.where(dev > 0.5, logi_floor + 1, logi_floor)

    return logi

  def _normalized_ps(self, ps, vocab_size):
    ps = torch.round(ps).to(torch.int64)
    ps = torch.where(ps < vocab_size, ps, (vocab_size - 1) * torch.ones(ps.shape).to(torch.int64))
    ps = torch.where(ps >= 0, ps, torch.zeros(ps.shape).to(torch.int64))
    return ps

  def resize(self, image):
    h, w, _ = image.shape
    scale = 1024 / (max(w, h) + 1e-4)
    image = cv2.resize(image, (int(w * scale), int(h * scale)))
    image = cv2.copyMakeBorder(image, 0, 1024 - int(h * scale), 0, 1024 - int(w * scale), cv2.BORDER_CONSTANT,
                               value=[0, 0, 0])
    return image, scale

  def process(self, images, return_time=False):
    raise NotImplementedError

  def post_process(self, dets, meta, scale=1):
    raise NotImplementedError

  def merge_outputs(self, detections):
    raise NotImplementedError

  def debug(self, debugger, images, dets, output, scale=1):
    raise NotImplementedError

  def show_results(self, debugger, image, results):
    raise NotImplementedError

  def ps_convert_minmax(self, results):
    detection = {}
    for j in range(1, self.num_classes + 1):
      detection[j] = []
    for j in range(1, self.num_classes + 1):
      for bbox in results[j]:
        minx = min(bbox[0], bbox[2], bbox[4], bbox[6])
        miny = min(bbox[1], bbox[3], bbox[5], bbox[7])
        maxx = max(bbox[0], bbox[2], bbox[4], bbox[6])
        maxy = max(bbox[1], bbox[3], bbox[5], bbox[7])
        detection[j].append([minx, miny, maxx, maxy, bbox[-1]])
    for j in range(1, self.num_classes + 1):
      detection[j] = np.array(detection[j])
    return detection

  def run(self, opt, image_or_path_or_tensor, image_anno=None, meta=None):

    debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug == 3),
                        theme=self.opt.debugger_theme)

    pre_processed = False
    if isinstance(image_or_path_or_tensor, np.ndarray):
      image = image_or_path_or_tensor
    elif type(image_or_path_or_tensor) == type(''):
      image = cv2.imread(image_or_path_or_tensor)
    else:
      image = image_or_path_or_tensor['image'][0].numpy()
      pre_processed_images = image_or_path_or_tensor
      pre_processed = True

    if not opt.wiz_detect:
      batch = make_batch(opt, image_or_path_or_tensor, image_anno)

    detections = []
    hm = []
    corner_st = []
    if self.opt.demo != '':
      image_name = image_or_path_or_tensor.split('/')[-1]

    for scale in self.scales:

      if not pre_processed:
        images, meta = self.pre_process(image, scale, meta)

      else:
        images = pre_processed_images['images'][scale][0]
        meta = pre_processed_images['meta'][scale]
        meta = {k: v.numpy()[0] for k, v in meta.items()}

      images = images.to(self.opt.device)

      # torch.cuda.synchronize()

      if self.opt.wiz_detect:
        outputs, output, dets, corner_st_reg, forward_time, logi, cr, keep = self.process(images, image,
                                                                                          return_time=True)
      else:
        outputs, output, dets, corner_st_reg, forward_time, logi, cr, keep = self.process(images, image,
                                                                                          return_time=True, batch=batch)

      raw_dets = dets

      # torch.cuda.synchronize()

      if self.opt.debug >= 2:
        self.debug(debugger, images, dets, output, scale)

      dets, corner_st_reg = self.post_process(dets, meta, corner_st_reg, scale)
      # torch.cuda.synchronize()

      detections.append(dets)
      hm.append(keep)

    if self.opt.wiz_4ps or self.opt.wiz_2dpe:
      logi = logi + cr

    results = self.merge_outputs(detections)
    # torch.cuda.synchronize()

    slct_logi, slct_dets = self.filter(image_or_path_or_tensor, results, logi, raw_dets[:, :, :8])
    slct_dets = self._normalized_ps(slct_dets, 256)

    if self.opt.wiz_2dpe:
      if self.opt.wiz_stacking:
        _, slct_logi = self.processor(slct_logi, dets=slct_dets)
      else:
        slct_logi = self.processor(slct_logi, dets=slct_dets)
    else:
      if self.opt.wiz_stacking:
        _, slct_logi = self.processor(slct_logi)
      else:
        slct_logi = self.processor(slct_logi)

    slct_logi = self.process_logi(slct_logi)

    if self.opt.debug >= 1:
      self.show_results(debugger, image, results, corner_st_reg, image_name, slct_logi.squeeze())

    Results = self.ps_convert_minmax(results)
    return {'results': Results, '4ps': results, 'corner_st_reg': corner_st_reg, 'hm': hm}

class CtdetDetector(BaseDetector):
  def __init__(self, opt):
    super(CtdetDetector, self).__init__(opt)

  def process_logi(self, logi):
    logi_floor = logi.floor()
    dev = logi - logi_floor
    logi = torch.where(dev>0.5, logi_floor+1, logi_floor)
  
    logi0 = logi[:,:,0].unsqueeze(2)
    logi2 = logi[:,:,2].unsqueeze(2)

    logi_st = torch.cat((logi0, logi0, logi2, logi2), dim=2)
    logi = torch.where(logi<logi_st, logi_st, logi)
    return logi

  def process(self, images, origin, return_time=False, batch=None):
 
    with torch.no_grad():
      #outputs, feature_maps = self.model(images)

      outputs = self.model(images)
      output = outputs[-1]

      if batch is None :
        hm = output['hm'].sigmoid_()
        wh = output['wh']
        reg = output['reg'] if self.opt.reg_offset else None
        
      else:
        print('This results is generated from ground truth detection boxes.')
        hm = torch.Tensor(batch['hm']).unsqueeze(0).cuda()

        wh_ind = torch.tensor(batch['hm_ind']).expand(output['wh'].size(0), output['wh'].size(1), len(batch['hm_ind']))
        batchwh = torch.Tensor(batch['wh']).transpose(0,1).unsqueeze(0)
        wh = torch.zeros(size = output['wh'].size()).view(output['wh'].size(0), output['wh'].size(1), -1).scatter(2, wh_ind, batchwh)
        wh = wh.view(output['wh'].size(0), output['wh'].size(1), output['wh'].size(2), output['wh'].size(3)).cuda()
        #wh = wh + 2 * torch.rand(size = wh.shape).cuda()

        reg_ind = torch.tensor(batch['reg_ind']).expand(output['reg'].size(0), output['reg'].size(1), len(batch['reg_ind']))
        batchreg = torch.Tensor(batch['reg']).transpose(0,1).unsqueeze(0)
        reg = torch.zeros(size = output['reg'].size()).view(output['reg'].size(0), output['reg'].size(1), -1).scatter(2, reg_ind, batchreg)
        reg = reg.view(output['reg'].size(0), output['reg'].size(1), output['reg'].size(2), output['reg'].size(3)).cuda()
      
      st = output['st']
      ax = output['ax']
      cr = output['cr']
        
      if self.opt.flip_test:
        hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
        wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
        reg = reg[0:1] if reg is not None else None

      # torch.cuda.synchronize()
      forward_time = time.time()

      #return dets [bboxes, scores, clses]
    
      scores, inds, ys, xs, st_reg, corner_dict = corner_decode(hm[:,1:2,:,:], st, reg, K=int(self.opt.MK))
      dets, keep, logi, cr = ctdet_4ps_decode(hm[:,0:1,:,:], wh, ax, cr, corner_dict, reg=reg, K=self.opt.K, wiz_rev = self.opt.wiz_rev)
      corner_output = np.concatenate((np.transpose(xs.cpu()),np.transpose(ys.cpu()),np.array(st_reg.cpu()),np.transpose(scores.cpu())), axis=2)
     
      #logi = self.process_logi(logi)

    if return_time:
      return outputs, output, dets, corner_output, forward_time, logi, cr, keep#, overlayed_map
    else:
      return outputs, output, dets, logi, cr, keep#, corner_output

  def post_process(self, dets, meta, corner_st, scale=1):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    #return dets is list and what in dets is dict. key of dict is classes, value of dict is [bbox,score]
    if self.opt.upper_left:
      dets = ctdet_4ps_post_process_upper_left(
          dets.copy(), [meta['c']], [meta['s']],
          meta['out_height'], meta['out_width'], self.opt.num_classes)
    else:
      dets = ctdet_4ps_post_process(
          dets.copy(), [meta['c']], [meta['s']],
          meta['out_height'], meta['out_width'], self.opt.num_classes)
    corner_st = ctdet_corner_post_process(
        corner_st.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], self.opt.num_classes)
    for j in range(1, self.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 9)
      dets[0][j][:, :8] /= scale
    return dets[0],corner_st[0]

  def merge_outputs(self, detections):
    results = {}
    for j in range(1, self.num_classes + 1):
      results[j] = np.concatenate(
        [detection[j] for detection in detections], axis=0).astype(np.float32)
      if len(self.scales) > 1 or self.opt.nms:
         #soft_nms(results[j], Nt=0.5, method=2)
         results[j] = pnms(results[j],self.opt.thresh_min,self.opt.thresh_conf)
    scores = np.hstack(
      [results[j][:, 8] for j in range(1, self.num_classes + 1)])
    if len(scores) > self.max_per_image:
      kth = len(scores) - self.max_per_image
      thresh = np.partition(scores, kth)[kth]
      for j in range(1, self.num_classes + 1):
        keep_inds = (results[j][:, 8] >= thresh)
        results[j] = results[j][keep_inds]
    return results

  def debug(self, debugger, images, dets, output, scale=1):
    detection = dets.detach().cpu().numpy().copy()
    detection[:, :, :8] *= self.opt.down_ratio
    for i in range(1):
      img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
      img = ((img * self.std + self.mean) * 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
      debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
      for k in range(len(dets[i])):
        if detection[i, k, 8] > self.opt.center_thresh:
          debugger.add_4ps_coco_bbox(detection[i, k, :8], detection[i, k, -1],
                                 detection[i, k, 8], 
                                 img_id='out_pred_{:.1f}'.format(scale))

  def show_results(self, debugger, image, results, corner, image_name, logi=None):
    debugger.add_img(image, img_id='ctdet')
    m,n = corner.shape
    
    count = 0
 
    # fc = open(self.opt.output_dir + self.opt.demo_name +'/center/'+image_name+'.txt','w+') #bounding boxes saved
    # fv = open(self.opt.output_dir + self.opt.demo_name +'/corner/'+image_name+'.txt','w+')
    # fl = open(self.opt.output_dir + self.opt.demo_name +'/logi/'+image_name+'.txt','w+') #logic axis saved
    for j in range(1, self.num_classes + 1):
      k = 0
      for m in range(len(results[j])):
        bbox = results[j][m]
        k = k + 1
        if bbox[8] > self.opt.vis_thresh:
       
          if len(logi.shape) == 1:
            debugger.add_4ps_coco_bbox(bbox[:8], j-1, bbox[8], logi, show_txt=True, img_id='ctdet')
          else:
            debugger.add_4ps_coco_bbox(bbox[:8], j-1, bbox[8], logi[m,:], show_txt=True, img_id='ctdet')
          # for i in range(0,3):
            position_holder = 1
            # fc.write(str(bbox[2*i])+','+str(bbox[2*i+1])+';')
            # if not logi is None:
              # if len(lo//gi.shape) == 1:
                # fl.write(str(int(logi[i]))+',')
              # else:
                # fl.write(str(int(logi[m,:][i]))+',')
          # fc.write(str(bbox[6])+','+str(bbox[7])+'\n')
          #
          # if not logi is None:
          #   if len(logi.shape) == 1:
          #     fl.write(str(int(logi[3]))+'\n')
          #   else:
          #     fl.write(str(int(logi[m,:][3]))+'\n')

    #   if self.opt.vis_corner==1:
    #     for i in range(m):
    #       if corner[i,10] > self.opt.vis_thresh_corner:
    #         for w in range(0,4):
    #           position_holder = 1
    #           fv.write(str(corner[i,2*w])+','+str(corner[i,2*w+1])+';')
    #         fv.write(str(corner[i,8])+','+str(corner[i,9])+'\n')
    #         count+=1
    #
    # fc.close()
    # fv.close()
    debugger.save_all_imgs(image_name, self.opt.demo_dir)
      
 