import os
from typing import Union, Dict, Tuple, Optional
import numpy as np
import datetime
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from pytorch_lightning import LightningDataModule
import xarray as xr
import glob

class IceConcentrationDataset(TorchDataset):
    def __init__(self,
                 data: np.ndarray,
                 seq_len: int = 25,
                 sample_mode: str = "sequent",
                 stride: int = 12,
                 layout: str = "NTHWC",
                 in_len: int = 13,
                 out_len: int = 12):
        """
        Parameters
        ----------
        data : np.ndarray  # Shape (total_timesteps, height, width, channels)
        """
        self.data = data
        self.seq_len = seq_len
        self.sample_mode = sample_mode
        self.stride = stride
        self.layout = layout
        self.in_len = in_len
        self.out_len = out_len

        # Validate parameters
        if sample_mode != "sequent":
            raise NotImplementedError("Only sequent sample mode supported")

        self.total_samples = (len(data) - seq_len) // stride + 1

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.seq_len
        full_seq = self.data[start:end]  # (seq_len, H, W, C)

        # Split into input and target
        in_seq = full_seq[:self.in_len]
        out_seq = full_seq[self.in_len:]

        # Convert to tensor
        return {
            'input': torch.from_numpy(in_seq).float(),
            'target': torch.from_numpy(out_seq).float()
        }

    def collate_fn(self, batch):
        return {
            'input': torch.stack([item['input'] for item in batch], dim=0),
            'target': torch.stack([item['target'] for item in batch], dim=0)
        }

class IceConcentrationDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 seq_len: int = 25,
                 sample_mode: str = "sequent",
                 stride: int = 12,
                 batch_size: int = 1,
                 layout: str = "NTHWC",
                 in_len: int = 13,
                 out_len: int = 12,
                 output_type: type = np.float32,
                 preprocess: bool = True,
                 rescale_method: str = "01",
                 num_workers: int = 1,
                 subset_ratio: float = 0.2,
                 split_seed: int = 42):
        super().__init__()
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.sample_mode = sample_mode
        self.stride = stride
        self.batch_size = batch_size
        self.layout = layout
        self.in_len = in_len
        self.out_len = out_len
        self.output_type = output_type
        self.preprocess = preprocess
        self.rescale_method = rescale_method
        self.num_workers = num_workers
        self.subset_ratio = subset_ratio
        self.split_seed = split_seed

        # Data storage
        self.full_data = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # Load and preprocess data
        if self.full_data is None:
            # Load NetCDF files
            nc_files = glob.glob(os.path.join(self.data_dir, "*.nc"))
            combined_ds = xr.open_mfdataset(nc_files, combine="nested",
                                          concat_dim="time", chunks={"time": 100})

            # Create subset
            time_size = len(combined_ds.time)
            subset_size = int(time_size * self.subset_ratio)
            random_indices = np.random.choice(time_size, subset_size, replace=False)
            random_indices.sort()  # Maintain temporal order

            # Process data
            ice_conc = combined_ds.isel(time=random_indices)["ice_conc"]
            ice_conc = ice_conc.fillna(0).compute().data  # Handle NaNs
            ice_conc = np.clip(ice_conc, 0.0, 1.0)  # Normalize

            # Add channel dimension (T, H, W) -> (T, H, W, 1)
            self.full_data = ice_conc[..., np.newaxis].astype(self.output_type)

    def setup(self, stage: Optional[str] = None):
        if not self.train_dataset:
            # Split data into train/val/test (70/15/15)
            assert self.full_data.shape[1:3] == (140, 120), \
            f"Data has wrong spatial dimensions: {self.full_data.shape}"
            total_timesteps = len(self.full_data)
            train_size = int(0.7 * total_timesteps)
            val_size = int(0.15 * total_timesteps)

            # Create splits
            train_data = self.full_data[:train_size]
            val_data = self.full_data[train_size:train_size+val_size]
            test_data = self.full_data[train_size+val_size:]

            # Create datasets
            self.train_dataset = IceConcentrationDataset(
                train_data, self.seq_len, self.sample_mode, self.stride,
                self.layout, self.in_len, self.out_len
            )
            self.val_dataset = IceConcentrationDataset(
                val_data, self.seq_len, self.sample_mode, self.stride,
                self.layout, self.in_len, self.out_len
            )
            self.test_dataset = IceConcentrationDataset(
                test_data, self.seq_len, self.sample_mode, self.stride,
                self.layout, self.in_len, self.out_len
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.train_dataset.collate_fn,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.test_dataset.collate_fn
        )

    @property
    def num_train_samples(self):
        return len(self.train_dataset) if self.train_dataset else 0

    @property
    def num_val_samples(self):
        return len(self.val_dataset) if self.val_dataset else 0

    @property
    def num_test_samples(self):
        return len(self.test_dataset) if self.test_dataset else 0

    def get_input_output_shapes(self):
        """Returns (input_shape, output_shape) based on layout"""
        if self.layout == "NTHWC":
            return (self.in_len, 140, 120, 1), (self.out_len, 140, 120, 1)
        raise ValueError(f"Unsupported layout: {self.layout}")