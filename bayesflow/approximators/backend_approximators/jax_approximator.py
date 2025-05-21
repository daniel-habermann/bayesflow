import jax
import keras

from bayesflow.utils import filter_kwargs

from keras.src.backend.jax.trainer import JAXEpochIterator
from keras.src import callbacks as callbacks_module


class JAXApproximator(keras.Model):
    # noinspection PyMethodOverriding
    def compute_metrics(self, *args, **kwargs) -> dict[str, jax.Array]:
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
        aggregate=True,
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
            # Create an iterator that yields batches of
            # input/target data.
            epoch_iterator = JAXEpochIterator(
                x=x,
                y=y,
                sample_weight=sample_weight,
                batch_size=batch_size,
                steps_per_epoch=steps,
                shuffle=False,
                steps_per_execution=self.steps_per_execution,
            )

        self._symbolic_build(iterator=epoch_iterator)
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
        self._record_training_state_sharding_spec()

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

        self._jax_state_synced = True
        with epoch_iterator.catch_stop_iteration():
            for step, iterator in epoch_iterator:
                callbacks.on_test_batch_begin(step)

                total_steps += 1

                if self._jax_state_synced:
                    # The state may have been synced by a callback.
                    state = self._get_jax_state(
                        trainable_variables=True,
                        non_trainable_variables=True,
                        metrics_variables=True,
                        purge_model_variables=True,
                    )
                    self._jax_state_synced = False

                step_logs, state = self.test_function(state, iterator)
                (
                    trainable_variables,
                    non_trainable_variables,
                    metrics_variables,
                ) = state

                if aggregate:
                    logs = _aggregate_fn(logs, step_logs)
                else:
                    logs = step_logs

                # Setting _jax_state enables callbacks to force a state sync
                # if they need to.
                self._jax_state = {
                    # I wouldn't recommend modifying non-trainable model state
                    # during evaluate(), but it's allowed.
                    "trainable_variables": trainable_variables,
                    "non_trainable_variables": non_trainable_variables,
                    "metrics_variables": metrics_variables,
                }

                # Dispatch callbacks. This takes care of async dispatch.
                callbacks.on_test_batch_end(step, step_logs)

                if self.stop_evaluating:
                    break

        if aggregate:
            logs = _reduce_fn(logs, total_steps)

        # Reattach state back to model (if not already done by a callback).
        self.jax_state_sync()

        logs = self._get_metrics_result_or_logs(logs)
        callbacks.on_test_end(logs)
        self._jax_state = None
        if not use_cached_eval_dataset:
            # Only clear sharding if evaluate is not called from `fit`.
            self._clear_jax_state_sharding()
        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    def stateless_compute_metrics(
        self,
        trainable_variables: any,
        non_trainable_variables: any,
        metrics_variables: any,
        data: dict[str, any],
        stage: str = "training",
    ) -> (jax.Array, tuple):
        """
        Things we do for jax:
        1. Accept trainable variables as the first argument
            (can be at any position as indicated by the argnum parameter
             in autograd, but needs to be an explicit arg)
        2. Accept, potentially modify, and return other state variables
        3. Return just the loss tensor as the first value
        4. Return all other values in a tuple as the second value

        This ensures:
        1. The function is stateless
        2. The function can be differentiated with jax autograd
        """
        state_mapping = []
        state_mapping.extend(zip(self.trainable_variables, trainable_variables))
        state_mapping.extend(zip(self.non_trainable_variables, non_trainable_variables))
        state_mapping.extend(zip(self.metrics_variables, metrics_variables))

        # perform a stateless call to compute_metrics
        with keras.StatelessScope(state_mapping) as scope:
            kwargs = filter_kwargs(data | {"stage": stage}, self.compute_metrics)
            metrics = self.compute_metrics(**kwargs)

        # update variables
        non_trainable_variables = [scope.get_current_value(v) for v in self.non_trainable_variables]
        metrics_variables = [scope.get_current_value(v) for v in self.metrics_variables]

        return metrics["loss"], (metrics, non_trainable_variables, metrics_variables)

    def stateless_test_step(self, state: tuple, data: dict[str, any]) -> (dict[str, jax.Array], tuple):
        trainable_variables, non_trainable_variables, metrics_variables = state

        loss, aux = self.stateless_compute_metrics(
            trainable_variables, non_trainable_variables, metrics_variables, data=data, stage="validation"
        )
        metrics, non_trainable_variables, metrics_variables = aux

        metrics_variables = self._update_loss(loss, metrics_variables)

        state = trainable_variables, non_trainable_variables, metrics_variables
        return metrics, state

    def stateless_train_step(self, state: tuple, data: dict[str, any]) -> (dict[str, jax.Array], tuple):
        trainable_variables, non_trainable_variables, optimizer_variables, metrics_variables = state

        grad_fn = jax.value_and_grad(self.stateless_compute_metrics, has_aux=True)

        (loss, aux), grads = grad_fn(
            trainable_variables, non_trainable_variables, metrics_variables, data=data, stage="training"
        )
        metrics, non_trainable_variables, metrics_variables = aux

        trainable_variables, optimizer_variables = self.optimizer.stateless_apply(
            optimizer_variables, grads, trainable_variables
        )

        metrics_variables = self._update_loss(loss, metrics_variables)

        state = trainable_variables, non_trainable_variables, optimizer_variables, metrics_variables
        return metrics, state

    def test_step(self, *args, **kwargs):
        return self.stateless_test_step(*args, **kwargs)

    def train_step(self, *args, **kwargs):
        return self.stateless_train_step(*args, **kwargs)

    def _update_loss(self, loss: jax.Array, metrics_variables: any) -> any:
        # update the loss progress bar, and possibly metrics variables along with it
        state_mapping = list(zip(self.metrics_variables, metrics_variables))
        with keras.StatelessScope(state_mapping) as scope:
            self._loss_tracker.update_state(loss)

        metrics_variables = [scope.get_current_value(v) for v in self.metrics_variables]

        return metrics_variables
