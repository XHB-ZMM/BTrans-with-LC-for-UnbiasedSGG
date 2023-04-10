import torch
import json
import h5py
import numpy as np
from matplotlib.pyplot import imshow
from PIL import Image, ImageDraw
import pdb

image_file = json.load(open('/data/dataset/vg/image_data.json'))
vocab_file = json.load(open('/data/dataset/vg/VG-SGG-dicts-with-attri.json'))
data_file = h5py.File('/data/dataset/vg/VG-SGG-with-attri.h5', 'r')
# remove invalid image
corrupted_ims = [1592, 1722, 4616, 4617]
tmp = []
for item in image_file:
    if int(item['image_id']) not in corrupted_ims:
        tmp.append(item)
image_file = tmp
# load detected results
detected_origin_path = '/data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_BCL/output/predcls-BiBiTransformer-logits-compensation/inference/VG_stanford_filtered_with_attribute_test/'
detected_origin_result = torch.load(detected_origin_path + 'eval_results.pytorch')
detected_info = json.load(open(detected_origin_path + 'visual_info.json'))

# get image info by index
def get_info_by_idx(idx, det_input, thres=0.1):
    groundtruth = det_input['groundtruths'][idx]
    prediction = det_input['predictions'][idx]
    # image path
    img_path = detected_info[idx]['img_file']
    # img_path = '/data/dataset/vg/VG_100K/1061.jpg'
    # boxes
    boxes = prediction.bbox
    # object labels
    # pdb.set_trace()
    idx2label = vocab_file['idx_to_label']
    labels = ['{}-{}'.format(idx,idx2label[str(i)]) for idx, i in enumerate(groundtruth.get_field('labels').tolist())]
    pred_labels = ['{}-{}'.format(idx,idx2label[str(i)]) for idx, i in enumerate(prediction.get_field('pred_labels').tolist())]
    pred_scores = prediction.get_field('pred_scores').tolist()
    # groundtruth relation triplet
    idx2pred = vocab_file['idx_to_predicate']
    gt_rels = groundtruth.get_field('relation_tuple').tolist()
    gt_rels = [(labels[i[0]], idx2pred[str(i[2])], labels[i[1]]) for i in gt_rels]
    # prediction relation triplet
    pred_rel_pair = prediction.get_field('rel_pair_idxs').tolist()
    pred_rel_label = prediction.get_field('pred_rel_scores')
    pred_rel_label[:,0] = 0
    pred_rel_score, pred_rel_label = pred_rel_label.max(-1)
    # pdb.set_trace()
    # mask = pred_rel_score > thres
    # pred_rel_score = pred_rel_score[mask]
    # pred_rel_label = pred_rel_label[mask]

    # 按照三元组的分数，由高到低排序,预测的三元组也按照这个顺序排列
    triplet_score = [pred_rel_score[idx]*pred_scores[pair[0]]*pred_scores[pair[0]] for idx, pair in enumerate(pred_rel_pair)]
    sorted_indexes = sorted(range(len(triplet_score)), key=lambda i: triplet_score[i],reverse=True)
    pred_rels = [(pred_labels[i[0]], idx2pred[str(j)], pred_labels[i[1]]) for i, j in zip(pred_rel_pair, pred_rel_label.tolist())]
    pred_rels = [pred_rels[i] for i in sorted_indexes]
    triplet_score = [triplet_score[i] for i in sorted_indexes]
    pred_rel_score = triplet_score

    return img_path, boxes, labels, pred_labels, pred_scores, gt_rels, pred_rels, pred_rel_score, pred_rel_label


def draw_single_box(pic, box, color='red', draw_info=None):
    draw = ImageDraw.Draw(pic)
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    draw.rectangle(((x1, y1), (x2, y2)), outline=color)
    if draw_info:
        draw.rectangle(((x1, y1), (x1 + 50, y1 + 10)), fill=color)
        info = draw_info
        draw.text((x1, y1), info)


def print_list(name, input_list, scores):
    for i, item in enumerate(input_list):
        if scores == None:
            print(name + ' ' + str(i) + ': ' + str(item))
        else:
            print(name + ' ' + str(i) + ': ' + str(item) + '; score: ' + str(scores[i].item()))


def draw_image(img_path, boxes, labels, pred_labels, pred_scores, gt_rels, pred_rels, pred_rel_score, pred_rel_label,
               print_img=True,cand_idx=0):
    pic = Image.open(img_path)
    pic.save("/data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_BCL/tools/BTransLC/test_ori_" + str(cand_idx) +"_.png")
    num_obj = boxes.shape[0]
    for i in range(num_obj):
        info = pred_labels[i]
        draw_single_box(pic, boxes[i], draw_info=info)
        pic.save("/data/xuhongbo/xuhongbo.code/unbiased_sgg_xuhongbo_BCL/tools/BTransLC/test_det_" + str(cand_idx) +"_.png")

    # if print_img:
    #     display(pic)
    if print_img:
        print('*' * 50)
        print_list('gt_boxes', labels, None)
        print('*' * 50)
        print_list('gt_rels', gt_rels, None)
        print('*' * 50)
    # pdb.set_trace()
    print_list('pred_labels', pred_labels, pred_rel_score)
    print('*' * 50)
    print_list('pred_rels', pred_rels, pred_rel_score)
    print('*' * 50)

    return None


def show_selected(idx_list):
    for select_idx in idx_list:
        print(select_idx)
        draw_image(*get_info_by_idx(select_idx, detected_origin_result))


def show_all(start_idx, length):
    for cand_idx in range(start_idx, start_idx + length):
        print(f'Image {cand_idx}:')
        img_path, boxes, labels, pred_labels, pred_scores, gt_rels, pred_rels, pred_rel_score, pred_rel_label = get_info_by_idx(
            cand_idx, detected_origin_result)
        draw_image(img_path=img_path, boxes=boxes, labels=labels, pred_labels=pred_labels, pred_scores=pred_scores,
                   gt_rels=gt_rels, pred_rels=pred_rels, pred_rel_score=pred_rel_score, pred_rel_label=pred_rel_label,
                   print_img=True,cand_idx=cand_idx)

show_all(start_idx=21, length=10)
# show_selected([119, 967, 713, 5224, 19681, 25371])