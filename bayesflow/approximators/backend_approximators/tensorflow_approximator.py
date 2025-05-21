import keras
import tensorflow as tf

from bayesflow.utils import filter_kwargs

from keras.src.backend.tensorflow.trainer import TFEpochIterator
from keras.src import callbacks as callbacks_module


class TensorFlowApproximator(keras.Model):
    # noinspection PyMethodOverriding
    def compute_metrics(self, *args, **kwargs) -> dict[str, tf.Tensor]:
        # implemented by each respective architecture
        raise NotImplementedError

    def evaluate(
        self,
        x=None,
        y=None,
        batch_size=None,
        verbose="auto",
        sample_weight=None,
        steps=None,
        callbacks=None,
        return_dict=False,
        aggregate=False,
        **kwargs,
    ):
        self._assert_compile_called("evaluate")
        # TODO: respect compiled trainable state
        use_cached_eval_dataset = kwargs.pop("_use_cached_eval_dataset", False)
        if kwargs:
            raise ValueError(f"Arguments not recognized: {kwargs}")

        if use_cached_eval_dataset:
            epoch_iterator = self._eval_epoch_iterator
        else:
            # Create an iterator that yields batches of input/target data.
            epoch_iterator = TFEpochIterator(
                x=x,
                y=y,
                sample_weight=sample_weight,
                batch_size=batch_size,
                steps_per_epoch=steps,
                shuffle=False,
                distribute_strategy=self.distribute_strategy,
                steps_per_execution=self.steps_per_execution,
            )

        self._maybe_symbolic_build(iterator=epoch_iterator)
        epoch_iterator.reset()

        # Container that configures and calls callbacks.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=1,
                steps=epoch_iterator.num_batches,
                model=self,
            )

        self.make_test_function()
        self.stop_evaluating = False
        callbacks.on_test_begin()
        logs = {}
        total_steps = 0
        self.reset_metrics()

        def _aggregate_fn(_logs, _step_logs):
            if not _logs:
                return _step_logs

            return keras.tree.map_structure(keras.ops.add, _logs, _step_logs)

        def _reduce_fn(_logs, _total_steps):
            def _div(val):
                return val / _total_steps

            return keras.tree.map_structure(_div, _logs)

        with epoch_iterator.catch_stop_iteration():
            for step, iterator in epoch_iterator:
                callbacks.on_test_batch_begin(step)
                total_steps += 1

                step_logs = self.test_function(iterator)

                if aggregate:
                    logs = _aggregate_fn(logs, step_logs)
                else:
                    logs = step_logs

                callbacks.on_test_batch_end(step, step_logs)
                if self.stop_evaluating:
                    break

        if aggregate:
            logs = _reduce_fn(logs, total_steps)

        logs = self._get_metrics_result_or_logs(logs)
        callbacks.on_test_end(logs)

        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    def test_step(self, data: dict[str, any]) -> dict[str, tf.Tensor]:
        kwargs = filter_kwargs(data | {"stage": "validation"}, self.compute_metrics)
        return self.compute_metrics(**kwargs)

    def train_step(self, data: dict[str, any]) -> dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            kwargs = filter_kwargs(data | {"stage": "training"}, self.compute_metrics)
            metrics = self.compute_metrics(**kwargs)

        loss = metrics["loss"]

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self._loss_tracker.update_state(loss)

        return metrics
