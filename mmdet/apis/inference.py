import warnings
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
import pdb
from mmdet.core import get_classes
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
from mmcv.image import imread, imwrite
import cv2


def init_detector(config, checkpoint=None, device='cuda:0'):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    model = build_detector(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage(object):

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def inference_detector(model, img):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)

    return result


# TODO: merge this method with the one in BaseDetector
def show_result(img,
                result,
                class_names,
                score_thr=0.3,
                wait_time=0,
                show=True,
                out_file=None):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        wait_time (int): Value of waitKey param.
        show (bool, optional): Whether to show the image with opencv or not.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.

    Returns:
        np.ndarray or None: If neither `show` nor `out_file` is specified, the
            visualized image is returned, otherwise None is returned.
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    mmcv.imshow_det_bboxes(
        img,
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,
        show=show,
        wait_time=wait_time,
        out_file=out_file)
    if not (show or out_file):
        return img


def iou(bbox1, bbox2):
    overlap_left = max(bbox1[0], bbox2[0])
    overlap_right = min(bbox1[2], bbox2[2])
    overlap_bottom = max(bbox1[1], bbox2[1])
    overlap_top = min(bbox1[3], bbox2[3])
    if overlap_left >= overlap_right or overlap_top <= overlap_bottom:
        return 0
    else:
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        area_overlap = (overlap_right - overlap_left) * (overlap_top - overlap_bottom)
        return area_overlap / (area1 + area2 - area_overlap)


def iou_vector(bbox1, bbox2):
    select_indexes = np.where((bbox1[:, 0] < bbox2[2]) & (bbox1[:, 2] > bbox2[0]) &
                              (bbox1[:, 1] < bbox2[3]) & (bbox1[:, 3] > bbox2[1]))[0]
    if len(select_indexes) == 0:
        return -1, 0
    bbox1_selected = bbox1[select_indexes, :]
    area1 = (bbox1_selected[:, 2] - bbox1_selected[:, 0]) * (bbox1_selected[:, 3] - bbox1_selected[:, 1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    overlap_left = np.maximum(bbox1_selected[:, 0], bbox2[0])
    overlap_right = np.minimum(bbox1_selected[:, 2], bbox2[2])
    overlap_bottom = np.maximum(bbox1_selected[:, 1], bbox2[1])
    overlap_top = np.minimum(bbox1_selected[:, 3], bbox2[3])
    area_overlap = (overlap_right - overlap_left) * (overlap_top - overlap_bottom)
    max_index = select_indexes[np.argmax(area_overlap / (area1 + area2 - area_overlap))]
    max_value = np.max(area_overlap / (area1 + area2 - area_overlap))
    return max_index, max_value


def calc_map(map_iou, map_label):
    map_area = 0
    positive_iou = np.sort(map_iou[np.where(map_label == 1)[0]])
    negative_iou = np.sort(map_iou[np.where(map_label == 0)[0]])
    len_positive = len(positive_iou)
    for i in range(0, len_positive):
        len_negative_selected = len(np.where(negative_iou >= positive_iou[i])[0])
        map_area += (len_positive - i) / (len_negative_selected + len_positive - i)
    return map_area / len_positive


def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      class_names=None,
                      score_thr=0,
                      thickness=5,
                      font_scale=2,
                      show=True,
                      win_name='',
                      wait_time=0,
                      out_file=None):
    """Draw bboxes and class labels (with scores) on an image.
    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    img = imread(img)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    cm = plt.cm.get_cmap('RdYlBu')
    text_color = (255, 255, 255)

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        bbox_color = cm(label / 90.0)
        bbox_color = (int(bbox_color[0]*255), int(bbox_color[1]*255), int(bbox_color[2]*255))
        cv2.rectangle(
            img, left_top, right_bottom, bbox_color, thickness=thickness)
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        if len(bbox) > 4:
            label_text += '|{:.02f}'.format(bbox[-1])
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color, 3)

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)


def show_result_plus_acc(img, result, class_names, gt_bboxes, gt_labels, map_data,
                         score_thr=0.3, wait_time=0, show=True, out_file=None):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        gt_bboxes: ground truth (bboxes).
        gt_labels: ground truth (labels).
        map_data: data for map.
        score_thr (float): The threshold to visualize the bboxes and masks.
        wait_time (int): Value of waitKey param.
        show (bool, optional): Whether to show the image with opencv or not.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.

    Returns:
        np.ndarray or None: If neither `show` nor `out_file` is specified, the
            visualized image is returned, otherwise None is returned.
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    gt_bboxes = torch.cat((gt_bboxes, torch.ones(gt_bboxes.size()[0], 1)), 1).numpy()
    if type(gt_labels) is int:
        if out_file:
            gt_labels = np.array([0] * len(gt_bboxes))
            labels = np.array([0] * len(bboxes))
            imshow_det_bboxes(
                img.copy(),
                gt_bboxes,
                gt_labels,
                class_names=class_names,
                score_thr=score_thr,
                show=show,
                wait_time=wait_time,
                out_file=out_file + '_gt.jpg')
            imshow_det_bboxes(
                img,
                bboxes,
                labels,
                class_names=class_names,
                score_thr=score_thr,
                show=show,
                wait_time=wait_time,
                out_file=out_file + '.jpg')
        indexes = np.where(bboxes[:, -1] > score_thr)[0]
        bboxes = bboxes[indexes]
        map_iou = []
        for i in range(0, len(gt_bboxes)):
            match_index, max_iou = iou_vector(bboxes, gt_bboxes[i])
            if match_index > -1:
                map_iou.append(max_iou)
            else:
                map_iou.append(0)
        return 0, float(sum(map_iou)) / len(map_iou), 0, None
    else:
        gt_labels = gt_labels.numpy() - 1
    if out_file:
        imshow_det_bboxes(
            img.copy(),
            gt_bboxes,
            gt_labels,
            class_names=class_names,
            score_thr=score_thr,
            show=show,
            wait_time=wait_time,
            out_file=out_file + '_gt.jpg')
        imshow_det_bboxes(
            img,
            bboxes,
            labels,
            class_names=class_names,
            score_thr=score_thr,
            show=show,
            wait_time=wait_time,
            out_file=out_file + '.jpg')
    indexes = np.where(bboxes[:, -1] > score_thr)[0]
    bboxes = bboxes[indexes]
    labels = labels[indexes]
    iou_acc = 0
    class_acc = 0
    map_iou = []
    map_label = []
    flag0 = np.zeros(len(class_names))
    flag1 = np.zeros(len(class_names))
    for i in range(0, len(gt_labels)):
        if map_data[1][gt_labels[i]] is None:
            if flag1[gt_labels[i]] == 0:
                map_data[1][gt_labels[i]] = [np.array([gt_bboxes[i]])]
                flag1[gt_labels[i]] = 1
        else:
            if flag1[gt_labels[i]] == 0:
                map_data[1][gt_labels[i]].append(np.array([gt_bboxes[i]]))
                flag1[gt_labels[i]] = 1
            else:
                map_data[1][gt_labels[i]][-1] = np.concatenate((map_data[1][gt_labels[i]][-1], np.array([gt_bboxes[i]])))
        match_index, max_iou = iou_vector(bboxes, gt_bboxes[i])
        if match_index > -1:
            map_iou.append(max_iou)
            if labels[match_index] == gt_labels[i]:
                class_acc += 1
                iou_acc += max_iou
                map_label.append(1)
            else:
                map_label.append(0)
    for i in range(0, len(labels)):
        if map_data[0][labels[i]] is None:
            if flag0[labels[i]] == 0:
                map_data[0][labels[i]] = [np.array([bboxes[i]])]
                flag0[labels[i]] = 1
        else:
            if flag0[labels[i]] == 0:
                map_data[0][labels[i]].append(np.array([bboxes[i]]))
                flag0[labels[i]] = 1
            else:
                map_data[0][labels[i]][-1] = np.concatenate((map_data[0][labels[i]][-1], np.array([bboxes[i]])))
    for i in range(len(class_names)):
        if flag0[i] == 0 and flag1[i] == 1:
            if len(map_data[1][i]) == 1:
                map_data[1][i] = None
            else:
                map_data[1][i].pop()
        if flag0[i] == 1 and flag1[i] == 0:
            if map_data[1][i] is None:
                map_data[1][i] = [np.array([])]
            else:
                map_data[1][i].append(np.array([]))
    if class_acc == 0:
        iou_acc = 0
        iou_acc2 = 0
    else:
        iou_acc2 = iou_acc / len(gt_labels)
        iou_acc /= class_acc
        class_acc /= len(gt_labels)
    return class_acc, iou_acc, iou_acc2, map_data


def show_result_pyplot(img,
                       result,
                       class_names,
                       score_thr=0.3,
                       fig_size=(15, 10)):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.
    """
    img = show_result(
        img, result, class_names, score_thr=score_thr, show=False)
    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))


def test_acc(img, result, class_names, gt_bboxes, gt_labels,
                         score_thr=0.3, wait_time=0, show=True, out_file=None):
    #TODO: acc calculation, maybe map_data is not useful
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        gt_bboxes: ground truth (bboxes).
        gt_labels: ground truth (labels).
        map_data: data for map.
        score_thr (float): The threshold to visualize the bboxes and masks.
        wait_time (int): Value of waitKey param.
        show (bool, optional): Whether to show the image with opencv or not.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.

    Returns:
        np.ndarray or None: If neither `show` nor `out_file` is specified, the
            visualized image is returned, otherwise None is returned.
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    gt_bboxes = torch.cat((gt_bboxes, torch.ones(gt_bboxes.size()[0], 1)), 1).numpy()
    if type(gt_labels) is int:
        if out_file:
            gt_labels = np.array([0] * len(gt_bboxes))
            labels = np.array([0] * len(bboxes))
            imshow_det_bboxes(
                img.copy(),
                gt_bboxes,
                gt_labels,
                class_names=class_names,
                score_thr=score_thr,
                show=show,
                wait_time=wait_time,
                out_file=out_file + '_gt.jpg')
            imshow_det_bboxes(
                img,
                bboxes,
                labels,
                class_names=class_names,
                score_thr=score_thr,
                show=show,
                wait_time=wait_time,
                out_file=out_file + '.jpg')
        indexes = np.where(bboxes[:, -1] > score_thr)[0]
        bboxes = bboxes[indexes]
        map_iou = []
        for i in range(0, len(gt_bboxes)):
            match_index, max_iou = iou_vector(bboxes, gt_bboxes[i])
            if match_index > -1:
                map_iou.append(max_iou)
            else:
                map_iou.append(0)
        return 0, float(sum(map_iou)) / len(map_iou), 0, None
    else:
        gt_labels = gt_labels.numpy() - 1
    if out_file:
        imshow_det_bboxes(
            img.copy(),
            gt_bboxes,
            gt_labels,
            class_names=class_names,
            score_thr=score_thr,
            show=show,
            wait_time=wait_time,
            out_file=out_file + '_gt.jpg')
        imshow_det_bboxes(
            img,
            bboxes,
            labels,
            class_names=class_names,
            score_thr=score_thr,
            show=show,
            wait_time=wait_time,
            out_file=out_file + '.jpg')
    indexes = np.where(bboxes[:, -1] > score_thr)[0]
    bboxes = bboxes[indexes]
    labels = labels[indexes]

    for i in range(0, len(gt_labels)):
        match_index, max_iou = iou_vector(bboxes, gt_bboxes[i])
        if match_index > -1:
            if labels[match_index] == gt_labels[i]:
                return 1
    
    return 0
