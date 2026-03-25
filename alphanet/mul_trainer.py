import torch
from torch.optim import Adam, AdamW, RAdam
from torch_geometric.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import pytorch_lightning as pl

def expand_stress_components(stress_6):
    
    xx, yy, zz, yz, xz, xy = stress_6
    return torch.tensor([xx, xy, xz, xy, yy, yz, xz, yz, zz])

def reshape_stress_tensor(stress_tensor ):
    
    reshaped = stress_tensor.view(-1, 6)
    
    expanded = torch.stack([expand_stress_components(row) for row in reshaped])
    result = expanded.view(-1, 3)
    
    return result
class Trainer(pl.LightningModule):
    def __init__(self, config, model, train_dataset, valid_dataset, test_dataset):
        super().__init__()
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.target = config.data.target

        if config.energy_loss == 'mse':
            self.energy_loss = torch.nn.MSELoss()
        elif config.energy_loss == 'mae':
            self.energy_loss = torch.nn.L1Loss()
        else:
            raise ValueError(f'Unknown energy loss: {config.train.energy_loss}')

        if self.config.compute_forces:
            if config.force_loss == 'mse':
                self.force_loss = torch.nn.MSELoss()
            elif config.force_loss == 'mae':
                self.force_loss = torch.nn.L1Loss()
            else:
                raise ValueError(f'Unknown force loss: {config.train.force_loss}')

        if config.energy_metric == 'mse':
            self.energy_metric = torch.nn.MSELoss()
        elif config.energy_metric == 'mae':
            self.energy_metric = torch.nn.L1Loss()
        else:
            raise ValueError(f'Unknown energy metric: {config.train.energy_metric}')

        if self.config.compute_forces:
            if config.force_metric == 'mse':
                self.force_metric = torch.nn.MSELoss()
            elif config.force_metric == 'mae':
                self.force_metric = torch.nn.L1Loss()
            else:
                raise ValueError(f'Unknown force metric: {config.train.force_metric}')
        if self.config.compute_stress:
             if config.stress_loss == 'mse':
                self.stress_loss = torch.nn.MSELoss()
             elif config.stress_loss == 'mae':
                self.stress_loss = torch.nn.L1Loss()
             else:
                raise ValueError(f'Unknown force loss: {config.train.force_loss}')
        
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
       batch_data = batch
       batch_data.pos.requires_grad = True
       if self.config.model.use_pbc:
          model_outputs = self.model(batch_data.pos, batch_data.z, batch_data.batch, batch_data.natoms, batch_data.cell, "train")
       else:
            model_outputs = self.model(batch_data.pos, batch_data.z, batch_data.batch, batch_data.natoms, prefix =  "train")

       e_loss, f_loss, s_loss = 0.0, 0.0, 0.0

       energy = model_outputs[0]

       e_loss = self.energy_loss(energy, batch_data.y)
       if self.config.compute_forces:
           forces = model_outputs[1]
           f_loss = self.force_loss(forces, batch_data.force)
       if self.config.compute_stress:
           stress = model_outputs[2]
           s_loss = self.stress_loss(stress, reshape_stress_tensor(batch_data.stress).to(batch_data.pos.device))

       loss = (self.config.train.energy_coef * e_loss +
               self.config.train.force_coef * f_loss +
               self.config.train.stress_coef * s_loss)
       # Pass batch_size explicitly: Lightning cannot reliably infer it from a
       # PyG DataBatch because iterating the batch yields individual Data objects
       # whose tensors have mismatched shape[0] (n_atoms vs 1 for energy labels).
       n_graphs = batch_data.num_graphs
       self.log('train_loss', loss, prog_bar=True, batch_size=n_graphs)
       self.log('train_energy_loss', e_loss, prog_bar=True, batch_size=n_graphs)
       if self.config.compute_forces:
           self.log('train_force_loss', f_loss, prog_bar=True, batch_size=n_graphs)
       if self.config.compute_stress:
           self.log('train_stress_loss', s_loss, prog_bar=True, batch_size=n_graphs)

       return loss


    def validation_step(self, batch, batch_idx):
       with torch.enable_grad():
        batch_data = batch
        batch_data.pos.requires_grad = True
        if self.config.model.use_pbc:
          model_outputs =  self.model(batch_data.pos, batch_data.z, batch_data.batch, batch_data.natoms, batch_data.cell, "infer")
        else:
            model_outputs = self.model(batch_data.pos, batch_data.z, batch_data.batch, batch_data.natoms, prefix ="infer")
        e_loss, f_loss, s_loss = 0.0, 0.0, 0.0

        energy = model_outputs[0]
        e_loss = self.energy_loss(energy, batch_data.y)
        if self.config.compute_forces:
            forces = model_outputs[1]
            f_loss = self.force_loss(forces, batch_data.force)
        if self.config.compute_stress:
            stress = model_outputs[2]
            s_loss = self.stress_loss(stress, reshape_stress_tensor(batch_data.stress).to(batch_data.pos.device))

        loss = (self.config.train.energy_coef * e_loss +
                self.config.train.force_coef * f_loss +
                self.config.train.stress_coef * s_loss)
        n_graphs = batch_data.num_graphs
        self.log('val_loss', loss, prog_bar=True, batch_size=n_graphs)
        self.log('val_energy_loss', e_loss, prog_bar=True, batch_size=n_graphs)
        if self.config.compute_forces:
            self.log('val_force_loss', f_loss, prog_bar=True, batch_size=n_graphs)
        if self.config.compute_stress:
            self.log('val_stress_loss', s_loss, prog_bar=True, batch_size=n_graphs)

        return loss

    def test_step(self, batch, batch_idx):
       with torch.enable_grad():
        batch_data = batch
        batch_data.pos.requires_grad = True
        if self.config.model.use_pbc:
          model_outputs =  self.model(batch_data.pos, batch_data.z, batch_data.batch, batch_data.natoms, batch_data.cell, "infer")
        else:
            model_outputs =  self.model(batch_data.pos, batch_data.z, batch_data.batch, batch_data.natoms, prefix = "infer")
        e_loss, f_loss, s_loss = 0.0, 0.0, 0.0

        energy = model_outputs[0]
        e_loss = self.energy_loss(energy.squeeze(), batch_data.y)
        if self.config.compute_forces:
            forces = model_outputs[1]
            f_loss = self.force_loss(forces, batch_data.force)
        if self.config.compute_stress:
            stress = model_outputs[2]
            s_loss = self.stress_loss(stress, batch_data.stress)

        loss = (self.config.train.energy_coef * e_loss +
                self.config.train.force_coef * f_loss +
                self.config.train.stress_coef * s_loss)
        n_graphs = batch_data.num_graphs
        self.log('val_loss', loss, prog_bar=True, batch_size=n_graphs)
        self.log('val_energy_loss', e_loss, prog_bar=True, batch_size=n_graphs)
        if self.config.compute_forces:
            self.log('val_force_loss', f_loss, prog_bar=True, batch_size=n_graphs)
        if self.config.compute_stress:
            self.log('val_stress_loss', s_loss, prog_bar=True, batch_size=n_graphs)

        return loss

    def configure_optimizers(self):
       
        opt_name = self.config.train.optimizer.lower()
        lr = self.config.train.lr
        weight_decay = self.config.train.weight_decay

        if opt_name == 'adam':
            optimizer = Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == 'adamw':
            optimizer = AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == 'radam':
            optimizer = RAdam(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        if self.config.train.scheduler == 'steplr':
            scheduler = StepLR(optimizer, step_size=self.config.train.lr_decay_step_size, gamma=self.config.train.lr_decay_factor)
        elif self.config.train.scheduler == 'cosineannealinglr':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.config.train.epochs, eta_min=1e-8)
        else:
            raise ValueError(f'Unknown scheduler: {self.config.train.scheduler}')

        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.train.batch_size, shuffle=False, num_workers=self.config.train.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.config.train.vt_batch_size, num_workers=self.config.train.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.train.vt_batch_size, num_workers=self.config.train.num_workers)

