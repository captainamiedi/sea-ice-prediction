import warnings
from typing import Union, Dict
from shutil import copyfile
from copy import deepcopy
import inspect
import pickle
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor, Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from omegaconf import OmegaConf
import os
import argparse
from einops import rearrange
from pytorch_lightning import Trainer, seed_everything
from earthformer.config import cfg
from earthformer.utils.optim import SequentialLR, warmup_lambda
from earthformer.utils.utils import get_parameter_names
# from earthformer.utils.checkpoint import pl_ckpt_to_pytorch_state_dict, s3_download_pretrained_ckpt
from earthformer.utils.layout import layout_to_in_out_slice
from earthformer.visualization.sevir.sevir_vis_seq import save_example_vis_results
from earthformer.metrics.sevir import SEVIRSkillScore
from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
from earthformer.datasets.sevir.sevir_torch_wrap import SEVIRLightningDataModule
# from earthformer.utils.apex_ddp import ApexDDPStrategy
import argparse
import sys
from datetime import datetime  # Add this import
from google.colab import drive
from dataset import IceConcentrationDataModule




_curr_dir = os.getcwd()
exps_dir = os.path.join(_curr_dir, "experiments")
pretrained_checkpoints_dir = cfg.pretrained_checkpoints_dir
pytorch_state_dict_name = "earthformer_ice_conc.pt"

class CuboidSEVIRPLModule(pl.LightningModule):

    def __init__(self,
                 total_num_steps: int,
                 oc_file: str = None,
                 save_dir: str = None):
        super(CuboidSEVIRPLModule, self).__init__()
        self._max_train_iter = total_num_steps
        if oc_file is not None:
            oc_from_file = OmegaConf.load(open(oc_file, "r"))
        else:
            oc_from_file = None
        oc = self.get_base_config(oc_from_file=oc_from_file)
        model_cfg = OmegaConf.to_object(oc.model)
        print("model:",model_cfg)
        num_blocks = len(model_cfg["enc_depth"])
        if isinstance(model_cfg["self_pattern"], str):
            enc_attn_patterns = [model_cfg["self_pattern"]] * num_blocks
        else:
            enc_attn_patterns = OmegaConf.to_container(model_cfg["self_pattern"])
        if isinstance(model_cfg["cross_self_pattern"], str):
            dec_self_attn_patterns = [model_cfg["cross_self_pattern"]] * num_blocks
        else:
            dec_self_attn_patterns = OmegaConf.to_container(model_cfg["cross_self_pattern"])
        if isinstance(model_cfg["cross_pattern"], str):
            dec_cross_attn_patterns = [model_cfg["cross_pattern"]] * num_blocks
        else:
            dec_cross_attn_patterns = OmegaConf.to_container(model_cfg["cross_pattern"])
        # print(f"num_merge = {num_merge}")
        # print("  out_dim_list:", self.out_dim_list)
        # print("  downsample_scale_list:", self.downsample_scale_list)
        # print("  lengths:", len(self.out_dim_list), len(self.downsample_scale_list))
        # ds = model_cfg["downsample"]
        # blocks = model_cfg["block_units"]
        # print(f"downsample list: {ds!r} (len={len(ds)})")
        # print(f"block_units    : {blocks!r} (len={len(blocks)})")
        self.torch_nn_module = CuboidTransformerModel(
            input_shape=model_cfg["input_shape"],
            target_shape=model_cfg["target_shape"],
            base_units=model_cfg["base_units"],
            block_units=model_cfg["block_units"],
            scale_alpha=model_cfg["scale_alpha"],
            enc_depth=model_cfg["enc_depth"],
            dec_depth=model_cfg["dec_depth"],
            enc_use_inter_ffn=model_cfg["enc_use_inter_ffn"],
            dec_use_inter_ffn=model_cfg["dec_use_inter_ffn"],
            dec_hierarchical_pos_embed=model_cfg["dec_hierarchical_pos_embed"],
            downsample=model_cfg["downsample"],
            downsample_type=model_cfg["downsample_type"],
            enc_attn_patterns=enc_attn_patterns,
            dec_self_attn_patterns=dec_self_attn_patterns,
            dec_cross_attn_patterns=dec_cross_attn_patterns,
            dec_cross_last_n_frames=model_cfg["dec_cross_last_n_frames"],
            dec_use_first_self_attn=model_cfg["dec_use_first_self_attn"],
            num_heads=model_cfg["num_heads"],
            attn_drop=model_cfg["attn_drop"],
            proj_drop=model_cfg["proj_drop"],
            ffn_drop=model_cfg["ffn_drop"],
            upsample_type=model_cfg["upsample_type"],
            ffn_activation=model_cfg["ffn_activation"],
            gated_ffn=model_cfg["gated_ffn"],
            norm_layer=model_cfg["norm_layer"],
            # global vectors
            num_global_vectors=model_cfg["num_global_vectors"],
            use_dec_self_global=model_cfg["use_dec_self_global"],
            dec_self_update_global=model_cfg["dec_self_update_global"],
            use_dec_cross_global=model_cfg["use_dec_cross_global"],
            use_global_vector_ffn=model_cfg["use_global_vector_ffn"],
            use_global_self_attn=model_cfg["use_global_self_attn"],
            separate_global_qkv=model_cfg["separate_global_qkv"],
            global_dim_ratio=model_cfg["global_dim_ratio"],
            # initial_downsample
            initial_downsample_type=model_cfg["initial_downsample_type"],
            initial_downsample_activation=model_cfg["initial_downsample_activation"],
            # initial_downsample_type=="stack_conv"
            initial_downsample_stack_conv_num_layers=model_cfg["initial_downsample_stack_conv_num_layers"],
            initial_downsample_stack_conv_dim_list=model_cfg["initial_downsample_stack_conv_dim_list"],
            initial_downsample_stack_conv_downscale_list=model_cfg["initial_downsample_stack_conv_downscale_list"],
            initial_downsample_stack_conv_num_conv_list=model_cfg["initial_downsample_stack_conv_num_conv_list"],
            # misc
            padding_type=model_cfg["padding_type"],
            z_init_method=model_cfg["z_init_method"],
            checkpoint_level=model_cfg["checkpoint_level"],
            pos_embed_type=model_cfg["pos_embed_type"],
            use_relative_pos=model_cfg["use_relative_pos"],
            self_attn_use_final_proj=model_cfg["self_attn_use_final_proj"],
            # initialization
            attn_linear_init_mode=model_cfg["attn_linear_init_mode"],
            ffn_linear_init_mode=model_cfg["ffn_linear_init_mode"],
            conv_init_mode=model_cfg["conv_init_mode"],
            down_up_linear_init_mode=model_cfg["down_up_linear_init_mode"],
            norm_init_mode=model_cfg["norm_init_mode"],
        )

        self.total_num_steps = total_num_steps
        if oc_file is not None:
            oc_from_file = OmegaConf.load(open(oc_file, "r"))
        else:
            oc_from_file = None
        oc = self.get_base_config(oc_from_file=oc_from_file)
        self.save_hyperparameters(oc)
        self.oc = oc
        # layout
        self.in_len = oc.layout.in_len
        self.out_len = oc.layout.out_len
        self.layout = oc.layout.layout
        # optimization
        self.max_epochs = oc.optim.max_epochs
        self.optim_method = oc.optim.method
        self.lr = oc.optim.lr
        self.wd = oc.optim.wd
        # lr_scheduler
        self.total_num_steps = total_num_steps
        self.lr_scheduler_mode = oc.optim.lr_scheduler_mode
        self.warmup_percentage = oc.optim.warmup_percentage
        self.min_lr_ratio = oc.optim.min_lr_ratio
        # logging
        self.save_dir = save_dir
        self.logging_prefix = oc.logging.logging_prefix
        # visualization
        self.train_example_data_idx_list = list(oc.vis.train_example_data_idx_list)
        self.val_example_data_idx_list = list(oc.vis.val_example_data_idx_list)
        self.test_example_data_idx_list = list(oc.vis.test_example_data_idx_list)
        self.eval_example_only = oc.vis.eval_example_only

        self.configure_save(cfg_file_path=oc_file)
        # evaluation
        self.metrics_list = oc.dataset.metrics_list
        self.threshold_list = oc.dataset.threshold_list
        self.metrics_mode = oc.dataset.metrics_mode
        self.valid_mse = torchmetrics.MeanSquaredError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        # New ice concentration specific metrics
        self.valid_ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0)
        self.valid_bias = torchmetrics.MeanMetric()
        # self.valid_csi = torchmetrics.MeanMetric()
        self.thresholds = torch.tensor([0.2, 0.5, 0.8])  # 20%, 50%, 80% concentration
        self.valid_csi = torchmetrics.MetricCollection({
            f'csi_{int(t*100)}%': torchmetrics.Accuracy(task='binary')
            for t in self.thresholds
        })
        self.valid_score = SEVIRSkillScore(
            mode=self.metrics_mode,
            seq_len=self.out_len,
            layout=self.layout,
            threshold_list=self.threshold_list,
            metrics_list=self.metrics_list,
            eps=1e-4,)
        self.test_mse = torchmetrics.MeanSquaredError()
        self.test_mae = torchmetrics.MeanAbsoluteError()
        self.test_score = SEVIRSkillScore(
            mode=self.metrics_mode,
            seq_len=self.out_len,
            layout=self.layout,
            threshold_list=self.threshold_list,
            metrics_list=self.metrics_list,
            eps=1e-4,)

    def configure_save(self, cfg_file_path=None):
        # Mount Google Drive (only once)
        drive.mount('/content/drive', force_remount=True)

        # Create save directory with timestamp
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        # self.save_dir = f"/content/drive/MyDrive/SavedModels/ice_concentration_{timestamp}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")  # Now uses imported class
        self.save_dir = f"/content/drive/MyDrive/SavedModels/ice_concentration_{timestamp}"

        os.makedirs(self.save_dir, exist_ok=True)
        self.scores_dir = os.path.join(self.save_dir, 'scores')
        os.makedirs(self.scores_dir, exist_ok=True)

        # Save config file
        if cfg_file_path is not None:
            cfg_file_target_path = os.path.join(self.save_dir, "cfg.yaml")
            copyfile(cfg_file_path, cfg_file_target_path)

        self.example_save_dir = os.path.join(self.save_dir, "examples")
        os.makedirs(self.example_save_dir, exist_ok=True)
        # self.save_dir = os.path.join(exps_dir, self.save_dir)
        # os.makedirs(self.save_dir, exist_ok=True)
        # self.scores_dir = os.path.join(self.save_dir, 'scores')
        # os.makedirs(self.scores_dir, exist_ok=True)
        # if cfg_file_path is not None:
        #     cfg_file_target_path = os.path.join(self.save_dir, "cfg.yaml")
        #     if (not os.path.exists(cfg_file_target_path)) or \
        #             (not os.path.samefile(cfg_file_path, cfg_file_target_path)):
        #         copyfile(cfg_file_path, cfg_file_target_path)
        # self.example_save_dir = os.path.join(self.save_dir, "examples")
        # os.makedirs(self.example_save_dir, exist_ok=True)
    def custom_save_checkpoint(self):
        """Custom checkpoint saving to Google Drive"""
        checkpoint_path = os.path.join(
            self.save_dir,
            "checkpoints",
            f"model_epoch_{self.current_epoch:03d}.ckpt"
        )

        # Save full model state
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizers().state_dict(),
            'lr_scheduler_state_dict': self.lr_schedulers().state_dict(),
            'loss': self.trainer.callback_metrics.get('valid_loss_epoch'),
            'config': OmegaConf.to_yaml(self.oc)
        }, checkpoint_path)

        print(f"Checkpoint saved to Google Drive: {checkpoint_path}")
    def get_base_config(self, oc_from_file=None):
        oc = OmegaConf.create()
        oc.dataset = self.get_dataset_config()
        oc.layout = self.get_layout_config()
        oc.optim = self.get_optim_config()
        oc.logging = self.get_logging_config()
        oc.trainer = self.get_trainer_config()
        oc.vis = self.get_vis_config()
        oc.model = self.get_model_config()
        if oc_from_file is not None:
            oc = OmegaConf.merge(oc, oc_from_file)
        return oc

    @staticmethod
    def get_dataset_config():
        # oc = OmegaConf.create()
        # oc.dataset_name = "sevir"
        # oc.img_height = 384
        # oc.img_width = 384
        # oc.in_len = 13
        # oc.out_len = 12
        oc = OmegaConf.create()
        oc.dataset_name = "ice_concentration"
        oc.img_height = 140  # From get_input_output_shapes()
        oc.img_width = 120   # Actual data dimensions
        oc.in_len = 13
        oc.out_len = 12
        oc.data_channels = 1  # Single channel for ice concentration
        oc.seq_len = 25
        oc.plot_stride = 2
        oc.interval_real_time = 5
        oc.sample_mode = "sequent"
        oc.stride = oc.out_len
        oc.layout = "NTHWC"
        oc.start_date = None
        oc.train_val_split_date = (2019, 1, 1)
        oc.train_test_split_date = (2019, 6, 1)
        oc.end_date = None
        oc.metrics_mode = "0"
        oc.metrics_list = ('csi', 'pod', 'sucr', 'bias')
        oc.threshold_list = (16, 74, 133, 160, 181, 219)
        return oc

    @classmethod
    def get_model_config(cls):
        # cfg = OmegaConf.create()
        # dataset_oc = cls.get_dataset_config()
        cfg = OmegaConf.create()
        dataset_oc = cls.get_dataset_config()
        cfg.input_shape = (
            dataset_oc.in_len,
            dataset_oc.img_height,
            dataset_oc.img_width,
            dataset_oc.data_channels
        )
        height = dataset_oc.img_height
        width = dataset_oc.img_width
        in_len = dataset_oc.in_len
        out_len = dataset_oc.out_len
        data_channels = 1
        # cfg.input_shape = (in_len, height, width, data_channels)
        cfg.target_shape = (out_len, height, width, data_channels)

        cfg.base_units = 64
        cfg.block_units = None # multiply by 2 when downsampling in each layer
        cfg.scale_alpha = 1.0

        cfg.enc_depth = [1, 1]
        cfg.dec_depth = [1, 1]
        cfg.enc_use_inter_ffn = True
        cfg.dec_use_inter_ffn = True
        cfg.dec_hierarchical_pos_embed = True

        cfg.downsample = 2
        cfg.downsample_type = "patch_merge"
        cfg.upsample_type = "upsample"

        cfg.num_global_vectors = 8
        cfg.use_dec_self_global = True
        cfg.dec_self_update_global = True
        cfg.use_dec_cross_global = True
        cfg.use_global_vector_ffn = True
        cfg.use_global_self_attn = False
        cfg.separate_global_qkv = False
        cfg.global_dim_ratio = 1

        cfg.self_pattern = 'axial'
        cfg.cross_self_pattern = 'axial'
        cfg.cross_pattern = 'cross_1x1'
        cfg.dec_cross_last_n_frames = None

        cfg.attn_drop = 0.1
        cfg.proj_drop = 0.1
        cfg.ffn_drop = 0.1
        cfg.num_heads = 4

        cfg.ffn_activation = 'gelu'
        cfg.gated_ffn = False
        cfg.norm_layer = 'layer_norm'
        cfg.padding_type = 'ignore'
        cfg.pos_embed_type = "t+hw"
        cfg.use_relative_pos = True
        cfg.self_attn_use_final_proj = True
        cfg.dec_use_first_self_attn = False

        cfg.z_init_method = 'zeros'
        cfg.checkpoint_level = 2
        # initial downsample and final upsample
        cfg.initial_downsample_type = "stack_conv"
        cfg.initial_downsample_activation = "leaky"
        cfg.initial_downsample_stack_conv_num_layers = 3
        cfg.initial_downsample_stack_conv_dim_list = [4, 16, 64]  # 3 layers
        cfg.initial_downsample_stack_conv_downscale_list = [2, 2, 1]  # 3 elements
        cfg.initial_downsample_stack_conv_num_conv_list = [2, 2, 2]   # 3 elements
        # initialization
        cfg.attn_linear_init_mode = "0"
        cfg.ffn_linear_init_mode = "0"
        cfg.conv_init_mode = "0"
        cfg.down_up_linear_init_mode = "0"
        cfg.norm_init_mode = "0"
        cfg.initial_downsample_kernel_size = (1, 3, 3)  # [T, H, W], C untouched
        cfg.initial_downsample_padding = (0, 1, 1)      # Pad H and W only
        return cfg

    @classmethod
    def get_layout_config(cls):
        oc = OmegaConf.create()
        dataset_oc = cls.get_dataset_config()
        oc.in_len = dataset_oc.in_len
        oc.out_len = dataset_oc.out_len
        oc.layout = dataset_oc.layout
        return oc

    @staticmethod
    def get_optim_config():
        oc = OmegaConf.create()
        oc.seed = None
        oc.total_batch_size = 32
        oc.micro_batch_size = 8

        oc.method = "adamw"
        oc.lr = 1E-3
        oc.wd = 1E-5
        oc.gradient_clip_val = 1.0
        oc.max_epochs = 100
        # scheduler
        oc.warmup_percentage = 0.2
        oc.lr_scheduler_mode = "cosine"  # Can be strings like 'linear', 'cosine', 'platue'
        oc.min_lr_ratio = 0.1
        oc.warmup_min_lr_ratio = 0.1
        # early stopping
        oc.early_stop = False
        oc.early_stop_mode = "min"
        oc.early_stop_patience = 20
        oc.save_top_k = 1
        return oc

    @staticmethod
    def get_logging_config():
        oc = OmegaConf.create()
        oc.logging_prefix = "SEVIR"
        oc.monitor_lr = True
        oc.monitor_device = False
        oc.track_grad_norm = -1
        oc.use_wandb = False
        return oc

    @staticmethod
    def get_trainer_config():
        oc = OmegaConf.create()
        oc.check_val_every_n_epoch = 1
        oc.log_step_ratio = 0.001  # Logging every 1% of the total training steps per epoch
        oc.precision = 32
        return oc

    @classmethod
    def get_vis_config(cls):
        oc = OmegaConf.create()
        dataset_oc = cls.get_dataset_config()
        oc.train_example_data_idx_list = [0, ]
        oc.val_example_data_idx_list = [80, ]
        oc.test_example_data_idx_list = [0, 80, 160, 240, 320, 400]
        oc.eval_example_only = False
        oc.plot_stride = dataset_oc.plot_stride
        return oc

    def configure_optimizers(self):
        # Configure the optimizer. Disable the weight decay for layer norm weights and all bias terms.
        decay_parameters = get_parameter_names(self.torch_nn_module, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [{
            'params': [p for n, p in self.torch_nn_module.named_parameters()
                       if n in decay_parameters],
            'weight_decay': self.oc.optim.wd
        }, {
            'params': [p for n, p in self.torch_nn_module.named_parameters()
                       if n not in decay_parameters],
            'weight_decay': 0.0
        }]

        if self.oc.optim.method == 'adamw':
            optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters,
                                          lr=self.oc.optim.lr,
                                          weight_decay=self.oc.optim.wd)
        else:
            raise NotImplementedError

        warmup_iter = int(np.round(self.oc.optim.warmup_percentage * self.total_num_steps))

        if self.oc.optim.lr_scheduler_mode == 'cosine':
            warmup_scheduler = LambdaLR(optimizer,
                                        lr_lambda=warmup_lambda(warmup_steps=warmup_iter,
                                                                min_lr_ratio=self.oc.optim.warmup_min_lr_ratio))
            cosine_scheduler = CosineAnnealingLR(optimizer,
                                                 T_max=(self.total_num_steps - warmup_iter),
                                                 eta_min=self.oc.optim.min_lr_ratio * self.oc.optim.lr)
            lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                        milestones=[warmup_iter])
            lr_scheduler_config = {
                'scheduler': lr_scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        else:
            raise NotImplementedError
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        """Required override for SequentialLR to work with Lightning"""
        scheduler.step()  # Step without metrics for schedule-based updates
    def set_trainer_kwargs(self, **kwargs):
        r"""
        Default kwargs used when initializing pl.Trainer
        """
        checkpoint_callback = ModelCheckpoint(
            monitor="valid_loss_epoch",
            dirpath=os.path.join(self.save_dir, "checkpoints"),
            filename="model-{epoch:03d}",
            save_top_k=self.oc.optim.save_top_k,
            save_last=True,
            mode="min",
        )
        callbacks = kwargs.pop("callbacks", [])
        assert isinstance(callbacks, list)
        for ele in callbacks:
            assert isinstance(ele, Callback)
        callbacks += [checkpoint_callback, ]
        if self.oc.logging.monitor_lr:
            callbacks += [LearningRateMonitor(logging_interval='step'), ]
        if self.oc.logging.monitor_device:
            callbacks += [DeviceStatsMonitor(), ]
        if self.oc.optim.early_stop:
            callbacks += [EarlyStopping(monitor="valid_loss_epoch",
                                        min_delta=0.0,
                                        patience=self.oc.optim.early_stop_patience,
                                        verbose=False,
                                        mode=self.oc.optim.early_stop_mode), ]

        logger = kwargs.pop("logger", [])
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=self.save_dir)
        csv_logger = pl_loggers.CSVLogger(save_dir=self.save_dir)
        logger += [tb_logger, csv_logger]
        if self.oc.logging.use_wandb:
            wandb_logger = pl_loggers.WandbLogger(project=self.oc.logging.logging_prefix,
                                                  save_dir=self.save_dir)
            logger += [wandb_logger, ]

        log_every_n_steps = max(1, int(self.oc.trainer.log_step_ratio * self.total_num_steps))
        trainer_init_keys = inspect.signature(Trainer).parameters.keys()
        ret = dict(
            callbacks=callbacks,
            # log
            logger=logger,
            log_every_n_steps=log_every_n_steps,
            track_grad_norm=self.oc.logging.track_grad_norm,
            # save
            default_root_dir=self.save_dir,
            # ddp
            accelerator="gpu",
            # strategy="ddp",
            # strategy=ApexDDPStrategy(find_unused_parameters=False, delay_allreduce=True),
            # optimization
            max_epochs=self.oc.optim.max_epochs,
            check_val_every_n_epoch=self.oc.trainer.check_val_every_n_epoch,
            gradient_clip_val=self.oc.optim.gradient_clip_val,
            # NVIDIA amp
            precision=self.oc.trainer.precision,
        )
        oc_trainer_kwargs = OmegaConf.to_object(self.oc.trainer)
        oc_trainer_kwargs = {key: val for key, val in oc_trainer_kwargs.items() if key in trainer_init_keys}
        ret.update(oc_trainer_kwargs)
        ret.update(kwargs)
        return ret

    @classmethod
    def get_total_num_steps(
            cls,
            num_samples: int,
            total_batch_size: int,
            epoch: int = None):
        r"""
        Parameters
        ----------
        num_samples:    int
            The number of samples of the datasets. `num_samples / micro_batch_size` is the number of steps per epoch.
        total_batch_size:   int
            `total_batch_size == micro_batch_size * world_size * grad_accum`
        """
        if epoch is None:
            epoch = cls.get_optim_config().max_epochs
        return int(epoch * num_samples / total_batch_size)

    @staticmethod
    def get_sevir_datamodule(dataset_oc,
                             micro_batch_size: int = 1,
                             num_workers: int = 8):
        # dm = SEVIRLightningDataModule(
        #     seq_len=dataset_oc["seq_len"],
        #     sample_mode=dataset_oc["sample_mode"],
        #     stride=dataset_oc["stride"],
        #     batch_size=micro_batch_size,
        #     layout=dataset_oc["layout"],
        #     output_type=np.float32,
        #     preprocess=True,
        #     rescale_method="01",
        #     verbose=False,
        #     # datamodule_only
        #     dataset_name=dataset_oc["dataset_name"],
        #     start_date=dataset_oc["start_date"],
        #     train_val_split_date=dataset_oc["train_val_split_date"],
        #     train_test_split_date=dataset_oc["train_test_split_date"],
        #     end_date=dataset_oc["end_date"],
        #     num_workers=num_workers,)
        dm = IceConcentrationDataModule(
            data_dir="/content/sea_ice_data",
            seq_len=25,          # Total sequence length (input + output)
            in_len=13,           # Input timesteps
            out_len=12,          # Output timesteps
            batch_size=8,
            layout="NTHWC",
            subset_ratio=0.2,
            num_workers=4
        )
        return dm

    @property
    def in_slice(self):
        if not hasattr(self, "_in_slice"):
            in_slice, out_slice = layout_to_in_out_slice(layout=self.layout,
                                                         in_len=self.in_len,
                                                         out_len=self.out_len)
            self._in_slice = in_slice
            self._out_slice = out_slice
        return self._in_slice

    @property
    def out_slice(self):
        if not hasattr(self, "_out_slice"):
            in_slice, out_slice = layout_to_in_out_slice(layout=self.layout,
                                                         in_len=self.in_len,
                                                         out_len=self.out_len)
            self._in_slice = in_slice
            self._out_slice = out_slice
        return self._out_slice

    def forward(self, in_seq, out_seq):
        """Updated to match new data format"""
        output = self.torch_nn_module(in_seq)
        loss = F.mse_loss(output, out_seq)
        return output, loss

    def training_step(self, batch, batch_idx):
        # """Process training batches"""
        # x = batch['input']  # Input sequence [B, T_in, H, W, C]
        # y = batch['target']  # Target sequence [B, T_out, H, W, C]

        # # Forward pass
        # y_hat, loss = self(x, y)

        # # Logging
        # self.log('train_loss', loss, prog_bar=True)
        # return loss
        x = batch['input']  # [B, T_in, H, W, C]
        y = batch['target']  # [B, T_out, H, W, C]

        # Forward pass
        y_hat, loss = self(x, y)

        # Additional training metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_mse', F.mse_loss(y_hat, y), prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx):
        # """Process validation batches"""
        # x = batch['input']
        # y = batch['target']

        # # Forward pass
        # y_hat, loss = self(x, y)

        # # Metrics
        # self.valid_mse.update(y_hat, y)
        # self.valid_mae.update(y_hat, y)
        # self.valid_score.update(y_hat, y)

        # # Logging
        # self.log('val_loss', loss, prog_bar=True)
        # return loss
        x = batch['input']
        y = batch['target']

        # Forward pass
        y_hat, loss = self(x, y)

        # Clamp values to valid range [0, 1]
        y_hat = torch.clamp(y_hat, 0.0, 1.0)
        y = torch.clamp(y, 0.0, 1.0)

        # Update metrics
        self.valid_mse(y_hat, y)
        self.valid_mae(y_hat, y)
        self.valid_ssim(y_hat, y)
        self.valid_bias(torch.mean(y_hat - y))

        # Calculate CSI for each threshold
        for t in self.thresholds:
            pred_binary = (y_hat > t).float()
            target_binary = (y > t).float()
            self.valid_csi[f'csi_{int(t.item()*100)}%'](pred_binary, target_binary)

        # Log losses
        self.log('val_loss', loss, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        # Log all metrics
        self.log('val_mse', self.valid_mse.compute(), prog_bar=True)
        self.log('val_mae', self.valid_mae.compute(), prog_bar=True)
        self.log('val_ssim', self.valid_ssim.compute(), prog_bar=True)
        self.log('val_bias', self.valid_bias.compute(), prog_bar=True)

        # Log CSI scores
        csi_scores = self.valid_csi.compute()
        for threshold, score in csi_scores.items():
            self.log(f'val_{threshold}', score, prog_bar=True)

        # Reset metrics
        self.valid_mse.reset()
        self.valid_mae.reset()
        self.valid_ssim.reset()
        self.valid_bias.reset()
        self.valid_csi.reset()

    def validation_epoch_end(self, outputs):
        valid_mse = self.valid_mse.compute()
        valid_mae = self.valid_mae.compute()
        self.log('valid_frame_mse_epoch', valid_mse,
                 prog_bar=True, on_step=False, on_epoch=True)
        self.log('valid_frame_mae_epoch', valid_mae,
                 prog_bar=True, on_step=False, on_epoch=True)
        self.valid_mse.reset()
        self.valid_mae.reset()
        valid_score = self.valid_score.compute()
        self.log("valid_loss_epoch", -valid_score["avg"]["csi"],
                 prog_bar=True, on_step=False, on_epoch=True)
        self.log_score_epoch_end(score_dict=valid_score, mode="val")
        self.valid_score.reset()
        self.save_score_epoch_end(score_dict=valid_score,
                                  mse=valid_mse,
                                  mae=valid_mae,
                                  mode="val")

    def test_step(self, batch, batch_idx):
        """Process test batches"""
        x = batch['input']
        y = batch['target']

        # Forward pass
        y_hat, loss = self(x, y)

        # Metrics
        self.test_mse.update(y_hat, y)
        self.test_mae.update(y_hat, y)
        self.test_score.update(y_hat, y)

        # Logging
        self.log('test_loss', loss, prog_bar=True)
        return loss
        # data_seq = batch['vil'].contiguous()
        # x = data_seq[self.in_slice]
        # y = data_seq[self.out_slice]
        # micro_batch_size = x.shape[self.layout.find("N")]
        # data_idx = int(batch_idx * micro_batch_size)
        # if not self.eval_example_only or data_idx in self.test_example_data_idx_list:
        #     y_hat, _ = self(x, y)
        #     self.save_vis_step_end(
        #         data_idx=data_idx,
        #         in_seq=x,
        #         target_seq=y,
        #         pred_seq=y_hat,
        #         mode="test"
        #     )
        #     if self.precision == 16:
        #         y_hat = y_hat.float()
        #     step_mse = self.test_mse(y_hat, y)
        #     step_mae = self.test_mae(y_hat, y)
        #     self.test_score.update(y_hat, y)
        #     self.log('test_frame_mse_step', step_mse,
        #              prog_bar=True, on_step=True, on_epoch=False)
        #     self.log('test_frame_mae_step', step_mae,
        #              prog_bar=True, on_step=True, on_epoch=False)
        return None

    def test_epoch_end(self, outputs):
        test_mse = self.test_mse.compute()
        test_mae = self.test_mae.compute()
        self.log('test_frame_mse_epoch', test_mse,
                 prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_frame_mae_epoch', test_mae,
                 prog_bar=True, on_step=False, on_epoch=True)
        self.test_mse.reset()
        self.test_mae.reset()
        test_score = self.test_score.compute()
        self.log_score_epoch_end(score_dict=test_score, mode="test")
        self.test_score.reset()
        self.save_score_epoch_end(score_dict=test_score,
                                  mse=test_mse,
                                  mae=test_mae,
                                  mode="test")

    def log_score_epoch_end(self, score_dict: Dict, mode: str = "val"):
        if mode == "val":
            log_mode_prefix = "valid"
        elif mode == "test":
            log_mode_prefix = "test"
        else:
            raise ValueError(f"Wrong mode {mode}. Must be 'val' or 'test'.")
        for metrics in self.metrics_list:
            for thresh in self.threshold_list:
                score_mean = np.mean(score_dict[thresh][metrics]).item()
                self.log(f"{log_mode_prefix}_{metrics}_{thresh}_epoch", score_mean)
            score_avg_mean = score_dict.get("avg", None)
            if score_avg_mean is not None:
                score_avg_mean = np.mean(score_avg_mean[metrics]).item()
                self.log(f"{log_mode_prefix}_{metrics}_avg_epoch", score_avg_mean)

    def save_score_epoch_end(self,
                             score_dict: Dict,
                             mse: Union[np.ndarray, float],
                             mae: Union[np.ndarray, float],
                             mode: str = "val"):
        assert mode in ["val", "test"], f"Wrong mode {mode}. Must be 'val' or 'test'."
        if self.local_rank == 0:
            save_dict = deepcopy(score_dict)
            save_dict.update(dict(mse=mse, mae=mae))
            if self.scores_dir is not None:
                save_path = os.path.join(self.scores_dir, f"{mode}_results_epoch_{self.current_epoch}.pkl")
                f = open(save_path, 'wb')
                pickle.dump(save_dict, f)
                f.close()

    def save_vis_step_end(
            self,
            data_idx: int,
            in_seq: torch.Tensor,
            target_seq: torch.Tensor,
            pred_seq: torch.Tensor,
            mode: str = "train"):
        r"""
        Parameters
        ----------
        data_idx:   int
            data_idx == batch_idx * micro_batch_size
        """
        if self.local_rank == 0:
            if mode == "train":
                example_data_idx_list = self.train_example_data_idx_list
            elif mode == "val":
                example_data_idx_list = self.val_example_data_idx_list
            elif mode == "test":
                example_data_idx_list = self.test_example_data_idx_list
            else:
                raise ValueError(f"Wrong mode {mode}! Must be in ['train', 'val', 'test'].")
            if data_idx in example_data_idx_list:
                save_example_vis_results(
                    save_dir=self.example_save_dir,
                    save_prefix=f'{mode}_epoch_{self.current_epoch}_data_{data_idx}',
                    in_seq=in_seq.detach().float().cpu().numpy(),
                    target_seq=target_seq.detach().float().cpu().numpy(),
                    pred_seq=pred_seq.detach().float().cpu().numpy(),
                    layout=self.layout,
                    plot_stride=self.oc.vis.plot_stride,
                    label=self.oc.logging.logging_prefix,
                    interval_real_time=self.oc.dataset.interval_real_time)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default='drive_save', type=str)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--cfg', default=None, type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--ckpt_name', default=None, type=str)
    parser.add_argument('--pretrained', action='store_true',  # ← Add this
                        help='Use pretrained model weights')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save checkpoint every N epochs')
    return parser

class DriveCheckpointCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # Save every N epochs
        if (trainer.current_epoch + 1) % trainer.datamodule.hparams.save_interval == 0:
            pl_module.custom_save_checkpoint()

def load_from_drive_checkpoint(checkpoint_path):
    drive.mount('/content/drive')
    checkpoint = torch.load(checkpoint_path)

    model = CuboidSEVIRPLModule.load_from_checkpoint(
        checkpoint_path,
        total_num_steps=checkpoint['config'].optim.total_num_steps,
        oc_file=checkpoint['config']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.optimizers().load_state_dict(checkpoint['optimizer_state_dict'])

    return model

def main():
    parser = get_parser()
    args, unknown = parser.parse_known_args()

    print("Parsed args:", args)
    print("Ignored unknown args:", unknown)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load configuration
    oc_from_file = OmegaConf.load("/content/config.yaml")
    dataset_oc = OmegaConf.to_object(oc_from_file.dataset)

    # Training parameters
    total_batch_size = oc_from_file.optim.total_batch_size
    micro_batch_size = oc_from_file.optim.micro_batch_size
    max_epochs = oc_from_file.optim.max_epochs
    seed = oc_from_file.optim.seed

    # Seed and data setup
    seed_everything(seed, workers=True)
    dm = CuboidSEVIRPLModule.get_sevir_datamodule(
        dataset_oc=dataset_oc,
        micro_batch_size=micro_batch_size,
        num_workers=8
    )
    dm.prepare_data()
    dm.setup()

    # Gradient accumulation
    accumulate_grad_batches = total_batch_size // (micro_batch_size * args.gpus)
    total_num_steps = CuboidSEVIRPLModule.get_total_num_steps(
        epoch=max_epochs,
        num_samples=dm.num_train_samples,
        total_batch_size=total_batch_size,
    )

    # Model setup
    pl_module = CuboidSEVIRPLModule(
        total_num_steps=total_num_steps,
        save_dir=args.save,
        oc_file=args.cfg
    )

    # Trainer configuration with Google Drive checkpointing
    trainer_kwargs = pl_module.set_trainer_kwargs(
        devices=args.gpus,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=[
            DriveCheckpointCallback(),
            LearningRateMonitor(),
            DeviceStatsMonitor()
        ]
    )
    trainer = Trainer(**trainer_kwargs)

    # Training/Testing workflow
    if args.pretrained:
        # Load pretrained weights with safety
        from omegaconf import DictConfig
        pretrained_ckpt_name = pytorch_state_dict_name
        ckpt_path = os.path.join(pretrained_checkpoints_dir, pretrained_ckpt_name)

        with torch.serialization.safe_globals([DictConfig] + list(torch.serialization.safe_globals)):
            state_dict = torch.load(ckpt_path,
                                  map_location=torch.device("cpu"),
                                  weights_only=True)

        pl_module.torch_nn_module.load_state_dict(state_dict)
        trainer.test(model=pl_module, datamodule=dm)

    elif args.test:
        # Testing mode
        assert args.ckpt_name, "Checkpoint name required for testing"
        ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
        trainer.test(model=pl_module, datamodule=dm, ckpt_path=ckpt_path)

    else:
        # Training mode with optional resume
        ckpt_path = None
        if args.ckpt_name:
            ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
            if not os.path.exists(ckpt_path):
                warnings.warn(f"Checkpoint {ckpt_path} not found, starting from scratch")
                ckpt_path = None

        trainer.fit(
            model=pl_module,
            datamodule=dm,
            ckpt_path=ckpt_path
        )

        # Final test with best checkpoint
        trainer.test(
            datamodule=dm,
            ckpt_path="best"
        )

if __name__ == "__main__":
    main()