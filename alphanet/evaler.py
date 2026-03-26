import torch
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from alphanet.models.model import AlphaNetWrapper
class Evaluator:
    def __init__(self, model_path, config, device='cpu'):
        self.device = torch.device(device)
        self.config = config
        self.model = None
        self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        

    def load_model(self, model_path):
        les_config = getattr(self.config, 'les', None)
        self.model = AlphaNetWrapper(self.config.model, les_config=les_config)
        checkpoint = torch.load(model_path)

        self.model.load_state_dict(checkpoint)

    @staticmethod
    def _warn_skip(split_name, quantity, reason):
        print(f"[alpha-eval] Skip {split_name} {quantity}: {reason}.")

    @staticmethod
    def _get_diag_limits(pred_batches, target_batches, quantity):
        if not pred_batches or not target_batches:
            raise RuntimeError(
                f"No valid {quantity} samples found across Train/Validation/Test splits."
            )

        combined = torch.cat(
            [
                torch.cat(target_batches, dim=0).reshape(-1),
                torch.cat(pred_batches, dim=0).reshape(-1),
            ],
            dim=0,
        )
        return combined.min().item(), combined.max().item()


    def plot_energy_parity(self, train_loader, val_loader, test_loader, plots_dir=None, disable=False):
        datasets = {'Train': train_loader, 'Validation': val_loader, 'Test': test_loader}
        colors = {'Train': '#1f77b4', 'Validation': '#ff7f0e', 'Test': '#2ca02c'}
        all_preds_energy = []
        all_targets_energy = []

        plt.figure(figsize=(12, 8))
        plt.grid(True, linestyle='--', alpha=0.5)

        for name, loader in datasets.items():
            preds_energy_batches = []
            targets_energy_batches = []

            for batch_data in tqdm(loader, disable=disable):
                batch_data = batch_data.to(self.device)
                batch_data.pos.requires_grad = True 
                if self.config.model.use_pbc:
                   model_outputs =  self.model(batch_data.pos, batch_data.z, batch_data.batch, batch_data.natoms, batch_data.cell, "infer")
                else:
                  model_outputs = self.model(batch_data.pos, batch_data.z, batch_data.batch, batch_data.natoms, prefix ="infer")
                
                energy, _, _ = model_outputs
                
                energy = energy.squeeze()
                preds_energy_batches.append((energy / batch_data.natoms).detach().cpu().reshape(-1))
                targets_energy_batches.append((batch_data.y.cpu() / batch_data.natoms.cpu()).reshape(-1))

            if not preds_energy_batches or not targets_energy_batches:
                self._warn_skip(name, "energy parity", "empty dataset")
                continue

            preds_energy = torch.cat(preds_energy_batches, dim=0)
            targets_energy = torch.cat(targets_energy_batches, dim=0)

            if preds_energy.numel() == 0 or targets_energy.numel() == 0:
                self._warn_skip(name, "energy parity", "no collected samples")
                continue

            deviation = torch.abs(preds_energy - targets_energy)
            threshold = 100 * torch.sqrt(torch.mean((preds_energy - targets_energy) ** 2)).item()
            mask_deviation = deviation < threshold
            mask_nan = ~torch.isnan(preds_energy) & ~torch.isnan(targets_energy)
            mask = mask_deviation & mask_nan
            preds_energy_filtered = preds_energy[mask]
            targets_energy_filtered = targets_energy[mask]

            if preds_energy_filtered.numel() == 0 or targets_energy_filtered.numel() == 0:
                self._warn_skip(name, "energy parity", "no valid samples after filtering")
                continue

            all_preds_energy.append(preds_energy_filtered)
            all_targets_energy.append(targets_energy_filtered)
            energy_mae_filtered = torch.mean(torch.abs(preds_energy_filtered - targets_energy_filtered)).item()
            energy_rmse_filtered = torch.sqrt(torch.mean((preds_energy_filtered - targets_energy_filtered) ** 2)).item()
            plt.scatter(
                targets_energy_filtered.cpu().numpy(),
                preds_energy_filtered.cpu().numpy(),
                alpha=0.7,
                label=f'{name}: MAE={energy_mae_filtered:.4f}, RMSE={energy_rmse_filtered:.4f}',
                color=colors[name],
                s=20
            )

        min_energy, max_energy = self._get_diag_limits(
            all_preds_energy,
            all_targets_energy,
            "energy",
        )
        plt.plot([min_energy, max_energy], [min_energy, max_energy], 'k--', lw=2, label='Ideal')
        plt.xlabel('True Energy per Atom', fontsize=17)
        plt.ylabel('Predicted Energy per Atom', fontsize=17)
        plt.title('Energy Parity Plot', fontsize=19)
        plt.legend(fontsize=17, loc='upper left', bbox_to_anchor=(0.1, 0.95), ncol=1, framealpha=0.8)
        plt.tight_layout()

        if plots_dir is None:
            plots_dir = './plots'
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, 'energy_parity_plot_combined.png'), dpi=1000)
        plt.close()

    def plot_force_parity(self, train_loader, val_loader, test_loader, plots_dir=None, disable=False):
        datasets = {'Train': train_loader, 'Validation': val_loader, 'Test': test_loader}
        colors = {'Train': '#1f77b4', 'Validation': '#ff7f0e', 'Test': '#2ca02c'}
        all_preds_force = []
        all_targets_force = []

        plt.figure(figsize=(12, 8))
        plt.grid(True, linestyle='--', alpha=0.5)

        for name, loader in datasets.items():
            preds_force_batches = []
            targets_force_batches = []

            for batch_data in tqdm(loader, disable=disable):
                batch_data = batch_data.to(self.device)
                batch_data.pos.requires_grad = True 
                if self.config.model.use_pbc:
                   model_outputs =  self.model(batch_data.pos, batch_data.z, batch_data.batch, batch_data.natoms, batch_data.cell, "infer")
                else:
                   model_outputs = self.model(batch_data.pos, batch_data.z, batch_data.batch, batch_data.natoms, prefix ="infer")
                if self.config.compute_forces:
                    _, force,_ = model_outputs

                    if torch.sum(torch.isnan(force)) != 0:
                        mask = ~torch.isnan(force) & ~torch.isnan(batch_data.force)
                        force = force[mask].reshape((-1, 3))
                        target_force = batch_data.force[mask].reshape((-1, 3))
                    else:
                        target_force = batch_data.force

                    preds_force_batches.append(force.detach().cpu())
                    targets_force_batches.append(target_force.cpu())

            if not preds_force_batches or not targets_force_batches:
                self._warn_skip(name, "force parity", "empty dataset")
                continue

            preds_force = torch.cat(preds_force_batches, dim=0)
            targets_force = torch.cat(targets_force_batches, dim=0)

            if preds_force.numel() == 0 or targets_force.numel() == 0:
                self._warn_skip(name, "force parity", "no collected samples")
                continue

            deviation = torch.abs(preds_force - targets_force)
            threshold = 100 * torch.sqrt(torch.mean((preds_force - targets_force) ** 2)).item()
            mask_deviation = deviation < threshold
            mask_nan = ~torch.isnan(preds_force) & ~torch.isnan(targets_force)
            mask = mask_deviation & mask_nan
            preds_force_filtered = preds_force[mask]
            targets_force_filtered = targets_force[mask]

            if preds_force_filtered.numel() == 0 or targets_force_filtered.numel() == 0:
                self._warn_skip(name, "force parity", "no valid samples after filtering")
                continue

            all_preds_force.append(preds_force_filtered.reshape(-1))
            all_targets_force.append(targets_force_filtered.reshape(-1))
            force_mae_filtered = torch.mean(torch.abs(preds_force_filtered - targets_force_filtered)).item()
            force_rmse_filtered = torch.sqrt(torch.mean((preds_force_filtered - targets_force_filtered) ** 2)).item()

            plt.scatter(
                targets_force_filtered.cpu().numpy(),
                preds_force_filtered.cpu().numpy(),
                alpha=0.7,
                label=f'{name}: MAE={force_mae_filtered:.4f}, RMSE={force_rmse_filtered:.4f}',
                color=colors[name],
                s=20
            )

        min_force, max_force = self._get_diag_limits(
            all_preds_force,
            all_targets_force,
            "force",
        )
        plt.plot([min_force, max_force], [min_force, max_force], 'k--', lw=2, label='Ideal')

        plt.xlabel('True Force', fontsize=17)
        plt.ylabel('Predicted Force', fontsize=17)
        plt.title('Force Parity Plot', fontsize=19)
        plt.legend(fontsize=17, loc='upper left', bbox_to_anchor=(0.1, 0.95), ncol=1, framealpha=0.8)
        plt.tight_layout()

        if plots_dir is None:
            plots_dir = './plots'
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, 'force_parity_plot_combined.png'), dpi=500)
        plt.close()

    def evaluate(self, data_path, plots_dir=None, disable=False):
        train_loader, val_loader, test_loader = self.load_data(data_path)
        self.plot_energy_parity(train_loader, val_loader, test_loader, plots_dir, disable)
        self.plot_force_parity(train_loader, val_loader, test_loader, plots_dir, disable)

if __name__ == '__main__':
    model_path = 'path_to_your_model.ckpt'  
    data_path = 'path_to_your_data'       
    plots_dir = './plots'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    evaluator = Evaluator(model_path, device)
    evaluator.evaluate(data_path, plots_dir)
