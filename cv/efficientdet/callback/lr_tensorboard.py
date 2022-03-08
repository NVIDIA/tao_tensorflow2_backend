import tensorflow as tf


class LRTensorBoard(tf.keras.callbacks.Callback):

    def __init__(self, log_dir, **kwargs):
        super().__init__(**kwargs)
        self.summary_writer = tf.summary.create_file_writer(log_dir)
        self.steps_before_epoch = 0
        self.steps_in_epoch = 0

    @property
    def global_steps(self):
        """The current 1-indexed global step."""
        return self.steps_before_epoch + self.steps_in_epoch

    def on_batch_end(self, batch, logs=None):
        self.steps_in_epoch = batch + 1

        lr = self.model.optimizer.lr(self.global_steps)
        with self.summary_writer.as_default():
            summary = tf.summary.scalar('learning_rate', lr, self.global_steps)

    def on_epoch_end(self, epoch, logs=None):
        self.steps_before_epoch += self.steps_in_epoch
        self.steps_in_epoch = 0

    def on_train_end(self, logs=None):
        self.summary_writer.flush()