import pickle
import random
import logging
from contextlib import nullcontext

import cv2
import torch
import numpy as np
from tqdm import tqdm, trange
import torch.profiler as profiler


class Runner:
    def __init__(
        self,
        cfg,
        exp,
        device,
        resume=False,
        view=None,
        deterministic=False,
        profile=False,
    ):
        self.cfg = cfg
        self.exp = exp
        self.device = device
        self.resume = resume
        self.view = view
        self.profile = profile
        self.logger = logging.getLogger(__name__)

        # Fix seeds
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        random.seed(cfg["seed"])

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def get_profiler_context(self):
        if self.profile:
            context = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=10, warmup=5, active=10),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    self.exp.get_tensorboard_path()
                ),
                record_shapes=True,
                profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
                with_stack=True,
            )
        else:
            context = nullcontext()
        return context

    def train(self):
        self.exp.train_start_callback(self.cfg)
        starting_epoch = 1
        model = self.cfg.get_model()
        model = model.to(self.device)
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
        # optimizer = self.cfg.get_optimizer(model.parameters())
        optimizer = self.cfg.get_optimizer(params_to_update)
        scheduler = self.cfg.get_lr_scheduler(optimizer)
        if self.resume:
            last_epoch, model, optimizer, scheduler = self.exp.load_last_train_state(
                model, optimizer, scheduler
            )
            starting_epoch = last_epoch + 1
        max_epochs = self.cfg["epochs"]
        train_loader = self.get_train_dataloader()
        loss_parameters = self.cfg.get_loss_parameters()
        with self.get_profiler_context() as profiler:
            for epoch in trange(
                starting_epoch,
                max_epochs + 1,
                initial=starting_epoch - 1,
                total=max_epochs,
            ):
                self.exp.epoch_start_callback(epoch, max_epochs)
                model.train()
                pbar = tqdm(train_loader)
                for i, (images, labels, _) in enumerate(pbar):
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    # Forward pass
                    outputs = model(images, **self.cfg.get_train_parameters())
                    loss, loss_dict_i = model.loss(outputs, labels, **loss_parameters)

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Scheduler step (iteration based)
                    scheduler.step()

                    # Profiler step (in case it is enabled)
                    if self.profile:
                        profiler.step()

                    # Log
                    postfix_dict = {
                        key: float(value) for key, value in loss_dict_i.items()
                    }
                    postfix_dict["lr"] = optimizer.param_groups[0]["lr"]
                    self.exp.iter_end_callback(
                        epoch,
                        max_epochs,
                        i,
                        len(train_loader),
                        loss.item(),
                        postfix_dict,
                    )
                    postfix_dict["loss"] = loss.item()
                    pbar.set_postfix(ordered_dict=postfix_dict)
                self.exp.epoch_end_callback(
                    epoch, max_epochs, model, optimizer, scheduler
                )

                # Validate
                if (epoch + 1) % self.cfg["val_every"] == 0:
                    self.eval(epoch, on_val=True)
        self.exp.train_end_callback()

    def eval(self, epoch, on_val=False, save_predictions=False):
        model = self.cfg.get_model()
        model_path = self.exp.get_checkpoint_path(epoch)
        self.logger.info("Loading model %s", model_path)
        model.load_state_dict(self.exp.get_epoch_model(epoch), strict=False)
        model = model.to(self.device)
        model.eval()
        if on_val:
            dataloader = self.get_val_dataloader()
        else:
            dataloader = self.get_test_dataloader()
        test_parameters = self.cfg.get_test_parameters()
        predictions = []
        self.exp.eval_start_callback(self.cfg)
        with torch.no_grad():
            for idx, (images, _, _) in enumerate(tqdm(dataloader)):
                images = images.to(self.device)
                output = model(images, **test_parameters)
                prediction = model.decode(output, as_lanes=True)
                predictions.extend(prediction)
                if self.view:
                    img = (images[0, -1].cpu().permute(1, 2, 0).numpy() * 255).astype(
                        np.uint8
                    )
                    img, fp, fn = dataloader.dataset.draw_annotation(
                        idx, img=img, pred=prediction[0]
                    )
                    if self.view == "mistakes" and fp == 0 and fn == 0:
                        continue
                    cv2.imshow("pred", img)
                    cv2.waitKey(0)

        if save_predictions:
            with open("predictions.pkl", "wb") as handle:
                pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.exp.eval_end_callback(dataloader.dataset.dataset, predictions, epoch)

    def get_train_dataloader(self):
        train_dataset = self.cfg.get_dataset("train")
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            worker_init_fn=self._worker_init_fn_,
        )
        return train_loader

    def get_test_dataloader(self):
        test_dataset = self.cfg.get_dataset("test")
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=self.cfg["batch_size"] if not self.view else 1,
            shuffle=False,
            num_workers=8,
            worker_init_fn=self._worker_init_fn_,
        )
        return test_loader

    def get_val_dataloader(self):
        val_dataset = self.cfg.get_dataset("val")
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=False,
            num_workers=8,
            worker_init_fn=self._worker_init_fn_,
        )
        return val_loader

    @staticmethod
    def _worker_init_fn_(_):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2**32 - 1
        random.seed(torch_seed)
        np.random.seed(np_seed)
