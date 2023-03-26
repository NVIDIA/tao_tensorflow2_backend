# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
"""EfficientDet standalone evaluation script."""
import logging
import os
from mpi4py import MPI
import tensorflow as tf

from nvidia_tao_tf2.common.decorators import monitor_status
from nvidia_tao_tf2.common.hydra.hydra_runner import hydra_runner
import nvidia_tao_tf2.common.logging.logging as status_logging
import nvidia_tao_tf2.common.no_warning # noqa pylint: disable=W0611
from nvidia_tao_tf2.common.utils import update_results_dir

from nvidia_tao_tf2.cv.efficientdet.config.default_config import ExperimentConfig
from nvidia_tao_tf2.cv.efficientdet.dataloader import dataloader, datasource
from nvidia_tao_tf2.cv.efficientdet.processor.postprocessor import EfficientDetPostprocessor
from nvidia_tao_tf2.cv.efficientdet.utils import coco_metric, label_utils
from nvidia_tao_tf2.cv.efficientdet.utils import helper, hparams_config
from nvidia_tao_tf2.cv.efficientdet.utils.config_utils import generate_params_from_cfg
from nvidia_tao_tf2.cv.efficientdet.utils.horovod_utils import (
    initialize, is_main_process, get_world_size, get_rank)
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level='INFO')
logger = logging.getLogger(__name__)


@monitor_status(name='efficientdet', mode='evaluation')
def run_experiment(cfg):
    """Run evaluation."""
    MODE = 'eval'
    # Parse and update hparams
    config = hparams_config.get_detection_config(cfg.model.name)
    config.update(generate_params_from_cfg(config, cfg, mode=MODE))

    # Set up dataloader
    eval_sources = datasource.DataSource(
        cfg.dataset.val_tfrecords,
        cfg.dataset.val_dirs)
    eval_dl = dataloader.CocoDataset(
        eval_sources,
        is_training=False,
        max_instances_per_image=config.max_instances_per_image)
    eval_dataset = eval_dl(
        config.as_dict(),
        batch_size=cfg.evaluate.batch_size)

    num_samples = (cfg.evaluate.num_samples + get_world_size() - 1) // get_world_size()
    num_samples = (num_samples + cfg.evaluate.batch_size - 1) // cfg.evaluate.batch_size
    cfg.evaluate.num_samples = num_samples
    eval_dataset = eval_dataset.shard(get_world_size(), get_rank()).take(num_samples)

    # Load model from graph json
    model = helper.load_model(cfg.evaluate.checkpoint, cfg, MODE, is_qat=cfg.train.qat)
    # Set up postprocessor
    postpc = EfficientDetPostprocessor(config)
    label_map = label_utils.get_label_map(cfg.evaluate.label_map)
    evaluator = coco_metric.EvaluationMetric(
        filename=cfg.dataset.val_json_file, label_map=label_map)

    @tf.function
    def eval_model_fn(images, labels):
        cls_outputs, box_outputs = model(images, training=False)
        detections = postpc.generate_detections(
            cls_outputs, box_outputs,
            labels['image_scales'],
            labels['source_ids'])

        def transform_detections(detections):
            # A transforms detections in [id, x1, y1, x2, y2, score, class]
            # form to [id, x, y, w, h, score, class]."""
            return tf.stack([
                detections[:, :, 0],
                detections[:, :, 1],
                detections[:, :, 2],
                detections[:, :, 3] - detections[:, :, 1],
                detections[:, :, 4] - detections[:, :, 2],
                detections[:, :, 5],
                detections[:, :, 6],
            ], axis=-1)

        tf.numpy_function(
            evaluator.update_state,
            [labels['groundtruth_data'], transform_detections(detections)], [])

    evaluator.reset_states()
    # evaluate all images.
    pbar = tf.keras.utils.Progbar(num_samples)
    for i, (images, labels) in enumerate(eval_dataset):
        eval_model_fn(images, labels)
        if is_main_process():
            pbar.update(i)

    # gather detections from all ranks
    evaluator.gather()

    if is_main_process():
        # compute the final eval results.
        metrics = evaluator.result()
        metric_dict = {}
        for i, name in enumerate(evaluator.metric_names):
            metric_dict[name] = metrics[i]

        if label_map:
            print("=============")
            print("Per class AP ")
            print("=============")
            for i, cid in enumerate(sorted(label_map.keys())):
                name = f'AP_{label_map[cid]}'
                metric_dict[name] = metrics[i + len(evaluator.metric_names)]
                print(f'{name}: {metric_dict[name]:.03f}')
        for k, v in metric_dict.items():
            status_logging.get_status_logger().kpi[k] = float(v)
    MPI.COMM_WORLD.Barrier()  # noqa pylint: disable=I1101


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="eval", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Wrapper function for EfficientDet evaluation."""
    cfg = update_results_dir(cfg, 'evaluate')
    initialize(cfg, logger, training=False)
    run_experiment(cfg)


if __name__ == '__main__':
    main()
