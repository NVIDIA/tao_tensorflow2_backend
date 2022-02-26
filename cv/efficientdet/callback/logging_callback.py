import tensorflow as tf

class LoggingCallback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        print("Iter: {}".format(batch))
        for var in self.model.variables:
            # if 'dense' in var.name:
            #   continue
            print("Var: {} {}".format(var.name, var.value))
            try:
                slot = self.model.optimizer.get_slot(var, "average")
                print("Avg: {}".format(slot))
            except KeyError as e:
                print("{} does not have ema average slot".format(var.name))