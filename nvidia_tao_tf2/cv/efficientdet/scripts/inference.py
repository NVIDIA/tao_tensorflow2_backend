# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""EfficientDet standalone inference."""
import logging
import os
import tensorflow as tf
from tensorflow.python.util import deprecation

from nvidia_tao_tf2.common.decorators import monitor_status
from nvidia_tao_tf2.common.hydra.hydra_runner import hydra_runner
import nvidia_tao_tf2.common.no_warning # noqa pylint: disable=W0611
from nvidia_tao_tf2.common.utils import update_results_dir

from nvidia_tao_tf2.cv.efficientdet.config.default_config import ExperimentConfig
from nvidia_tao_tf2.cv.efficientdet.inferencer import inference
from nvidia_tao_tf2.cv.efficientdet.utils import helper, hparams_config, label_utils
from nvidia_tao_tf2.cv.efficientdet.utils.config_utils import generate_params_from_cfg
from nvidia_tao_tf2.cv.efficientdet.utils.horovod_utils import initialize
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ["TF_CPP_VMODULE"] = 'non_max_suppression_op=0,generate_box_proposals_op=0,executor=0'
supported_img_format = ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level='INFO')
logger = logging.getLogger(__name__)


def get_label_dict(label_txt):
    """Create label dict from txt file."""
    with open(label_txt, 'r', encoding='utf-8') as f:
        labels = f.readlines()
        return {i + 1: label[:-1] for i, label in enumerate(labels)}


def batch_generator(iterable, batch_size=1):
    """Load a list of image paths in batches.

    Args:
        iterable: a list of image paths
        n: batch size
    """
    total_len = len(iterable)
    for ndx in range(0, total_len, batch_size):
        yield iterable[ndx:min(ndx + batch_size, total_len)]


@monitor_status(name='efficientdet', mode='inference')
def infer_tlt(cfg):
    """Launch EfficientDet TLT model Inference."""
    # disable_eager_execution()
    tf.autograph.set_verbosity(0)
    # Parse and update hparams
    MODE = 'infer'
    config = hparams_config.get_detection_config(cfg.model.name)
    config.update(generate_params_from_cfg(config, cfg, mode=MODE))
    params = config.as_dict()

    # Parse label map
    label_id_mapping = {}
    if cfg.inference.label_map:
        logger.info("Parsing label map...")
        if str(cfg.inference.label_map).endswith('.yaml'):
            label_id_mapping = label_utils.get_label_map(cfg.inference.label_map)
        else:
            label_id_mapping = get_label_dict(cfg.inference.label_map)

    # Load model from graph json
    logger.info('Loading model from: %s', cfg.inference.checkpoint)
    model = helper.load_model(cfg.inference.checkpoint, cfg, MODE, is_qat=cfg.train.qat)
    infer_model = inference.InferenceModel(model, config.image_size, params,
                                           cfg.inference.batch_size,
                                           label_id_mapping=label_id_mapping,
                                           min_score_thresh=cfg.inference.min_score_thresh,
                                           max_boxes_to_draw=cfg.inference.max_boxes_to_draw)
    imgpath_list = [os.path.join(cfg.inference.image_dir, imgname)
                    for imgname in sorted(os.listdir(cfg.inference.image_dir))
                    if os.path.splitext(imgname)[1].lower()
                    in supported_img_format]

    logger.info("Running inference...")
    for image_paths in batch_generator(imgpath_list, cfg.inference.batch_size):
        infer_model.visualize_detections(
            image_paths,
            cfg.inference.results_dir,
            cfg.inference.dump_label)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="inference", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig):
    """Wrapper function for EfficientDet inference."""
    cfg = update_results_dir(cfg, 'inference')
    initialize(cfg, logger, training=False)
    infer_tlt(cfg)


if __name__ == '__main__':
    main()
