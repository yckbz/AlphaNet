import click
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typing import Optional
from torch_geometric.data import DataLoader
from alphanet.data import get_pic_datasets
from alphanet.config import All_Config
from alphanet.evaler import Evaluator
import torch

console = Console()

def print_banner():
    """Display system initialization banner"""
    banner = Panel.fit(
        "[bold magenta]ALPHANET EVALUATION SUITE[/]",
        subtitle="[italic] Performance Analysis[/]",
        border_style="bright_blue",
        padding=(1, 2)
    )
    console.print(banner)

@click.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True),
              help="Path to configuration JSON file")
@click.option("--checkpoint", "-m", required=True, type=click.Path(exists=True),
              help="Path to trained model checkpoint")
@click.option("--device", default="cuda" if torch.cuda.is_available() else "cpu",
              show_default=True, help="Compute device (cuda/cpu)")
@click.option("--output-dir", type=click.Path(), default="./eval_results",
              show_default=True, help="Output directory for analysis artifacts")
@click.option("--data-root", default="dataset", show_default=True,
              help="Root directory containing datasets")
@click.option("--quiet", is_flag=True,
              help="Suppress progress indicators")
@click.option("--batch-size", type=int, default=None,
              help="Override batch size from config")
def evaluate(config: str, checkpoint: str, device: str, output_dir: str,
             data_root: str, quiet: bool, batch_size: Optional[int]):
    """Evaluate performance of molecular dynamics force field models
    
    Examples:
    
    \b
    # Basic usage
    alpha-eval --config config.json --checkpoint best_model.ckpt
    
    \b
    # Advanced usage
    alpha-eval -c config.json -m model.ckpt \\
        --device cuda:1 --output-dir ./analysis \\
        --data-root /mnt/datasets --batch-size 256
    """
    print_banner()
    
    # File validation
    validate_inputs(checkpoint, config)
    
    # Configuration setup
    config_obj = load_configuration(config)
    
    # Data pipeline initialization
    loaders = prepare_dataloaders(config_obj, data_root, batch_size)
    
    # Evaluation engine setup
    evaluator = initialize_evaluator(checkpoint, config_obj, device)
    
    # Execute analysis pipeline
    run_analysis(evaluator, loaders, output_dir, quiet)

def validate_inputs(ckpt_path: str, config_path: str):
    """Validate input file paths"""
    console.log("?? Validating input files...")
    
    required_files = {
        "Model checkpoint": Path(ckpt_path),
        "Config file": Path(config_path)
    }
    
    for name, path in required_files.items():
        if not path.exists():
            console.print(f"[red]{name} not found: {path}[/]")
            raise click.Abort()
        console.print(f"{name} validated: [cyan]{path}[/]")

def load_configuration(config_path: str) -> All_Config:
    """Load and display configuration"""
    console.log(" Loading configuration...")
    config = All_Config().from_json(config_path)
    
    # Display key parameters
    table = Table(title="Configuration Summary", show_header=False, box=None)
    table.add_column("Parameter", style="dim")
    table.add_column("Value")
    
    table.add_row("Dataset", config.data.dataset_name)
    table.add_row("Batch Size", str(config.train.batch_size))
    #table.add_row("Validation Split", f"{config.data.valid_size*100:.1f}%")
    
    console.print(Panel.fit(table, border_style="dim"))
    return config

def prepare_dataloaders(config: All_Config, data_root: str, batch_size: Optional[int]) -> dict:
    """Initialize data loaders"""
    console.log("Initializing data pipeline...")
    
    # Handle batch size override
    effective_batch_size = batch_size or config.train.batch_size
    
    # Load datasets
    train_set, valid_set, test_set = get_pic_datasets(
         root='dataset/', name=config.dataset_name,config = config
    )
    
    # Configure loader parameters
    loader_args = {
        "batch_size": effective_batch_size,
        "shuffle": False,
        "num_workers": config.train.num_workers
    }
    
    return {
        "train": DataLoader(train_set, **loader_args),
        "valid": DataLoader(valid_set, **loader_args),
        "test": DataLoader(test_set, **loader_args)
    }

def initialize_evaluator(ckpt_path: str, config: All_Config, device: str) -> Evaluator:
    """Initialize evaluation engine"""
    console.log(f"Initializing evaluator on [bold]{device.upper()}[/]")
    return Evaluator(
        model_path=ckpt_path,
        config=config,
        device=device
    )

def run_analysis(evaluator: Evaluator, loaders: dict, output_dir: str, quiet: bool):
    """Execute analysis workflow"""
    from pathlib import Path
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    split_sizes = {
        "Train": len(loaders["train"].dataset),
        "Validation": len(loaders["valid"].dataset),
        "Test": len(loaders["test"].dataset),
    }

    split_table = Table(title="Evaluation Splits", show_header=False, box=None)
    split_table.add_column("Split", style="dim")
    split_table.add_column("Samples")
    for split_name, split_size in split_sizes.items():
        split_table.add_row(split_name, str(split_size))
    console.print(Panel.fit(split_table, border_style="dim"))

    if all(size == 0 for size in split_sizes.values()):
        raise RuntimeError(
            "All evaluation splits are empty. Check train_size/valid_size/test_size or dataset configuration."
        )

    empty_splits = [name for name, size in split_sizes.items() if size == 0]
    if empty_splits:
        console.print(
            "[yellow]Warning:[/] Empty evaluation splits detected: "
            + ", ".join(empty_splits)
            + ". They will be skipped in parity plots."
        )
    
    console.log(f"Starting analysis: [cyan]{output_dir}[/]")
    
    # Energy metrics
    evaluator.plot_energy_parity(
        loaders["train"],
        loaders["valid"],
        loaders["test"],
        plots_dir=output_dir,
        disable=quiet
    )
    
    # Force metrics
    evaluator.plot_force_parity(
        loaders["train"],
        loaders["valid"],
        loaders["test"],
        plots_dir=output_dir,
        disable=quiet
    )
    
    console.print(f"\n[bold green]Analysis complete! Results saved to: [underline]{output_dir}[/][/]")

if __name__ == "__main__":
    evaluate()
