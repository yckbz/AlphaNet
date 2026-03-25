import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from alphanet.data import get_pic_datasets
from alphanet.models.model import AlphaNetWrapper
from alphanet.mul_trainer import Trainer

# Enable TF32 on Ampere+ GPUs (A100/A800/RTX30xx/RTX40xx) for faster matmul.
# TF32 has 10-bit mantissa vs float32's 23-bit, but the error is far smaller
# than DFT noise (~meV), so there is no practical impact on model accuracy.
torch.set_float32_matmul_precision('high')


def run_training(config1, runtime_config):

    train_dataset, valid_dataset, test_dataset = get_pic_datasets(
        root='dataset/', 
        name=config1.dataset_name, 
        config=config1
    )

    force_std = torch.std(train_dataset.data.force).item()
    if hasattr(train_dataset.data, 'y') and train_dataset.data.y is not None:
        energy_peratom = torch.sum(train_dataset.data.y).item() / torch.sum(train_dataset.data.natoms).item()
    else:
        energy_peratom = 0.0

    config1.a = force_std
    config1.b = energy_peratom

    les_config = getattr(config1, "les", None)
    model = AlphaNetWrapper(config1.model, les_config=les_config)
    
    if config1.dtype == "64":
        model = model.double()

 
    if runtime_config.get("finetune_path"):
        ft_path = runtime_config["finetune_path"]
        print(f"🔨 Finetuning mode: Loading weights from {ft_path}...")
        

        try:
            ckpt = torch.load(ft_path, map_location='cpu')
        except FileNotFoundError:
            raise FileNotFoundError(f"Finetune checkpoint not found at: {ft_path}")

      
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        new_state_dict = state_dict
        
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        
        if len(missing) > 0:
            print(f"   Warning: Missing keys ({len(missing)}): {missing[:3]} ...")
        if len(unexpected) > 0:
            print(f"   Warning: Unexpected keys ({len(unexpected)}): {unexpected[:3]} ...")
            
        print("   ✅ Weights loaded successfully. Optimizer states reset for finetuning.")
    else:
        print("🆕 Training from scratch (random initialization).")

  
    checkpoint_callback = ModelCheckpoint(
        dirpath=config1.train.save_dir,
        filename='{epoch}-{val_loss:.4f}-{val_energy_loss:.4f}-{val_force_loss:.4f}',
        save_top_k=-1,
        every_n_epochs=1,  
        save_on_train_epoch_end=True,  
        monitor='val_loss',
        mode='min'
    )
 
  
    trainer = pl.Trainer(
        devices=runtime_config["num_devices"],
        num_nodes=runtime_config["num_nodes"],
        strategy='ddp_find_unused_parameters_true', 
        accelerator="gpu" if runtime_config["num_devices"] > 0 and torch.cuda.is_available() else "cpu",
        max_epochs=config1.epochs,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
        gradient_clip_val=0.1,
        default_root_dir=config1.train.save_dir,
        accumulate_grad_batches=config1.accumulation_steps,
        limit_val_batches=100,
    )

    pl_module = Trainer(config1, model, train_dataset, valid_dataset, test_dataset)
    

    ckpt_path_arg = runtime_config["ckpt_path"] if runtime_config["resume"] else None
    
    if runtime_config["resume"] and ckpt_path_arg:
        print(f"🔄 Resuming training from checkpoint: {ckpt_path_arg}")
    
    trainer.fit(pl_module, ckpt_path=ckpt_path_arg)