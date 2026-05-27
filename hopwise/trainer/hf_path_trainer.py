# @Time   : 2025
# @Author : Giacomo Medda
# @Email  : giacomo.medda@unica.it

# UPDATE:
# @Time   : 2025
# @Author : Alessandro Soccol
# @Email  : alessandro.soccol@unica.it

from time import time

import torch
from transformers import DataCollatorForLanguageModeling, IntervalStrategy, Trainer, TrainerCallback

from hopwise.utils import (
    dict2str,
    early_stopping,
    get_gpu_usage,
    progress_bar,
    set_color,
)


class HFPathTrainer(Trainer):
    """A HuggingFace Trainer that integrates with Hopwise for training and evaluation."""

    def __init__(self, model, callbacks, train_data=None, args=None, tokenizer=None):
        tokenizer = tokenizer or train_data.dataset.tokenizer
        super().__init__(
            model=model,
            args=args,
            callbacks=None,
            train_dataset=train_data.dataset,
            eval_dataset="none",
            processing_class=tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        # Overwrite the callbacks to only use the HopwiseCallback
        self.callback_handler.callbacks = callbacks

    def evaluate(self, **kwargs):
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics=None)

        return self.control.metrics


class HopwiseCallback(TrainerCallback):
    """It handles the training and evaluation communication with the hopwise and HuggingFace trainers."""

    def __init__(
        self,
        hopwise_trainer,
        train_data=None,
        valid_data=None,
        verbose=True,
        saved=True,
        show_progress=False,
        callback_fn=None,
        model=None,
        model_name=None,
    ):
        self.model = model
        self.model_name = model_name
        self.hopwise_trainer = hopwise_trainer
        self.train_data = train_data
        self.valid_data = valid_data
        self.verbose = verbose
        self.saved = saved
        self.show_progress = show_progress
        self.callback_fn = callback_fn

    def on_train_begin(self, args, state, control, **kwargs):
        self.hopwise_trainer.eval_collector.train_data_collect(self.train_data)
        if self.hopwise_trainer.config["train_neg_sample_args"].get("dynamic", False):
            self.train_data.get_model(self.hopwise_trainer.model)
        self.valid_step = 0

        # Optional baseline evaluation before the first training epoch.
        # This evaluates the (pretrained) model on the validation set at
        # "epoch 0" so we can compare fine-tuning improvements against it.
        if self.valid_data is not None:
            epoch_idx = 0
            self.valid_start_time = time()
            valid_score, valid_result = self.hopwise_trainer._valid_epoch(
                self.valid_data, show_progress=self.show_progress
            )
            valid_end_time = time()
            valid_score_output = (
                set_color("epoch %d evaluating (baseline)", "green")
                + " ["
                + set_color("time", "blue")
                + ": %.2fs, "
                + set_color("valid_score", "blue")
                + ": %f]"
            ) % (epoch_idx, valid_end_time - self.valid_start_time, valid_score)
            valid_result_output = set_color("valid result (baseline)", "blue") + ": \n" + dict2str(valid_result)
            if self.verbose:
                self.hopwise_trainer.logger.info(valid_score_output)
                self.hopwise_trainer.logger.info(valid_result_output)
            self.hopwise_trainer.tensorboard.add_scalar("Valid_score", valid_score, epoch_idx)
            self.hopwise_trainer.wandblogger.log_metrics(
                {**valid_result, "valid_step": self.valid_step}, head="valid"
            )
            self.valid_step += 1

    def on_train_end(self, args, state, control, **kwargs):
        self.hopwise_trainer._add_hparam_to_tensorboard(self.hopwise_trainer.best_valid_score)
        return super().on_train_end(args, state, control, **kwargs)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.training_start_time = time()

        len_hf_dataloader = len(self.train_data.dataset)
        steps_in_epoch = len_hf_dataloader // self.hopwise_trainer.config["train_batch_size"]
        steps_in_epoch += int(len_hf_dataloader % self.hopwise_trainer.config["train_batch_size"] > 0)
        self.progress_bar = (
            progress_bar(
                total=steps_in_epoch,
                ncols=100,
                desc=set_color(f"Train {int(state.epoch):>5}", "magenta", progress=True),
            )
            if self.show_progress
            else range(steps_in_epoch)
        )

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.show_progress:
            self.progress_bar.close()
        training_end_time = time()
        # Retrieve training loss and other information
        if state.log_history:
            epoch_idx = state.epoch
            train_loss = state.log_history[-1].get("loss")
            self.hopwise_trainer.train_loss_dict[epoch_idx] = train_loss
            train_loss_output = self.hopwise_trainer._generate_train_loss_output(
                epoch_idx, self.training_start_time, training_end_time, train_loss
            )
            if self.verbose:
                self.hopwise_trainer.logger.info(train_loss_output)
            self.hopwise_trainer._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self.hopwise_trainer.wandblogger.log_metrics(
                {"epoch": epoch_idx, "train_loss": train_loss, "train_step": epoch_idx},
                head="train",
            )

        if self.hopwise_trainer.eval_step <= 0 or not self.valid_data:
            if self.saved:
                control.should_save = True
        elif epoch_idx % self.hopwise_trainer.eval_step == 0:
            control.should_evaluate = True
            self.valid_start_time = time()
        else:
            control.should_evaluate = False

        # update attentive-a
        if hasattr(self.model, "update_attentive_A"):
            with torch.no_grad():
                self.model.update_attentive_A()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        control.should_log = True
        if self.show_progress:
            self.progress_bar.update(1)
        if self.hopwise_trainer.gpu_available and self.show_progress:
            gpu_usage = get_gpu_usage(self.hopwise_trainer.device)
            self.progress_bar.set_postfix_str(set_color("GPU RAM: " + gpu_usage, "yellow"))

        # Log per-batch loss to W&B and capture initial loss at first step
        try:
            if state.log_history and isinstance(state.log_history[-1], dict):
                last_log = state.log_history[-1]
                if "loss" in last_log:
                    loss_val = float(last_log["loss"])
                    is_rank0 = self.hopwise_trainer.config["single_spec"] or self.hopwise_trainer.config.get("local_rank", 0) == 0
                    if is_rank0:
                        # Per-batch logging
                        self.hopwise_trainer.wandblogger.log_metrics(
                            {
                                "epoch": int(state.epoch or 0),
                                "global_step": state.global_step,
                                "batch_loss": loss_val,
                                # Ensure W&B uses a consistent step metric for all train/* logs
                                "train_step": state.global_step,
                            },
                            head="train",
                        )
                        # Initial loss at first global step
                        if state.global_step == 1:
                            self.hopwise_trainer.logger.info(set_color(f"Initial batch loss: {loss_val:.6f}", "blue"))
                            self.hopwise_trainer.tensorboard.add_scalar("Loss/Initial", loss_val, int(state.epoch or 0))
                            self.hopwise_trainer.wandblogger.log_metrics(
                                {
                                    "epoch": int(state.epoch or 0),
                                    "initial_loss": loss_val,
                                    "global_step": state.global_step,
                                    "train_step": state.global_step,
                                },
                                head="train",
                            )
        except Exception:
            # Swallow logging errors to avoid disrupting training
            pass

        if state.global_step >= state.max_steps:
            control.should_training_stop = True
            # Only save here for step-based strategies; epoch-based strategies
            # save in on_epoch_end, which runs after this and has the correct output_dir.
            if args.save_strategy == IntervalStrategy.STEPS:
                control.should_save = True

        return control

    def on_evaluate(self, args, state, control, **kwargs):
        epoch_idx = state.epoch
        valid_score, valid_result = self.hopwise_trainer._valid_epoch(
            self.valid_data, show_progress=self.show_progress
        )

        (
            self.hopwise_trainer.best_valid_score,
            self.hopwise_trainer.cur_step,
            stop_flag,
            update_flag,
        ) = early_stopping(
            valid_score,
            self.hopwise_trainer.best_valid_score,
            self.hopwise_trainer.cur_step,
            max_step=self.hopwise_trainer.stopping_step,
            bigger=self.hopwise_trainer.valid_metric_bigger,
        )
        valid_end_time = time()
        valid_score_output = (
            set_color("epoch %d evaluating", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
            + set_color("valid_score", "blue")
            + ": %f]"
        ) % (epoch_idx, valid_end_time - self.valid_start_time, valid_score)
        valid_result_output = set_color("valid result", "blue") + ": \n" + dict2str(valid_result)
        if self.verbose:
            self.hopwise_trainer.logger.info(valid_score_output)
            self.hopwise_trainer.logger.info(valid_result_output)
        self.hopwise_trainer.tensorboard.add_scalar("Valid_score", valid_score, epoch_idx)
        self.hopwise_trainer.wandblogger.log_metrics({**valid_result, "valid_step": self.valid_step}, head="valid")

        if not self.hopwise_trainer.valid_metric.startswith("eval_"):
            metric_to_check = f"eval_{self.hopwise_trainer.valid_metric}"
            control.metrics = {**valid_result, metric_to_check: valid_score}

        if update_flag:
            if self.saved:
                control.should_save = True
                self.hopwise_trainer._save_checkpoint(epoch_idx, verbose=self.verbose)

            self.hopwise_trainer.best_valid_result = valid_result

        if self.callback_fn:
            self.callback_fn(epoch_idx, valid_score)

        if stop_flag:
            stop_output = "Finished training, best eval result in epoch %d" % (
                epoch_idx - self.hopwise_trainer.cur_step * self.hopwise_trainer.eval_step
            )
            if self.verbose:
                self.hopwise_trainer.logger.info(stop_output)
            control.should_training_stop = True

        self.valid_step += 1

        return control
