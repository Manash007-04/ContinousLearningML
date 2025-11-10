"""
===================================================================
File 2: strategies.py - Training Strategies and Utilities
===================================================================
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from avalanche.training.strategies import Naive, EWC
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.storage import ClassBalancedBuffer
from clear_plugin import ClearReplayPlugin
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics, bwt_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class ExperimentConfig:
    """
    Centralized configuration for reproducibility and easy hyperparameter tuning.
    
    This dataclass pattern ensures all experimental settings are documented,
    version-controlled, and easily modified without touching the core code.
    """
    # Hardware Configuration
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    mixed_precision: bool = True  # Use automatic mixed precision for efficiency
    
    # Data Configuration
    n_experiences: int = 5  # Number of tasks in continual learning
    seed: int = 42  # Random seed for reproducibility
    
    # Training Hyperparameters
    train_epochs: int = 5  # Epochs per task
    batch_size: int = 64
    learning_rate: float = 1e-4  # Lower LR for pre-trained models
    weight_decay: float = 1e-4  # L2 regularization
    
    # Continual Learning Hyperparameters
    memory_size: int = 2000  # Replay buffer size
    ewc_lambda: float = 5000.0  # EWC regularization strength
    
    # LoRA Hyperparameters
    lora_r: int = 8  # Rank of LoRA decomposition
    lora_alpha: int = 16  # Scaling factor
    lora_dropout: float = 0.1  # Dropout for LoRA layers
    
    # Output Configuration
    output_dir: Path = Path("./cl_results")
    save_models: bool = False  # Whether to save model checkpoints
    
    def __post_init__(self):
        """Create output directory if it doesn't exist"""
        self.output_dir.mkdir(exist_ok=True, parents=True)


def get_evaluation_plugin():
    """
    Configure comprehensive evaluation metrics.
    
    Returns:
        EvaluationPlugin with multiple metrics for thorough analysis
    """
    interactive_logger = InteractiveLogger()
    
    eval_plugin = EvaluationPlugin(
        # Accuracy at different granularities
        accuracy_metrics(epoch=True, experience=True, stream=True),
        
        # Forgetting: how much performance drops on old tasks
        forgetting_metrics(experience=True, stream=True),
        
        # Backward Transfer: negative impact on past tasks
        bwt_metrics(experience=True, stream=True),
        
        loggers=[interactive_logger]
    )
    
    return eval_plugin


def create_strategy(strategy_name: str, model, config: ExperimentConfig, evaluator):
    """
    Factory function to instantiate different continual learning strategies.
    
    This design pattern makes it easy to add new strategies and ensures
    consistent hyperparameter usage across all methods.
    
    Args:
        strategy_name: Name of the CL strategy to use
        model: Neural network model
        config: Experiment configuration
        evaluator: Avalanche evaluation plugin
    
    Returns:
        Configured CL strategy ready for training
    """
    
    # Use AdamW optimizer (better for transformers than SGD)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Common arguments for all strategies
    base_kwargs = {
        'train_mb_size': config.batch_size,
        'train_epochs': config.train_epochs,
        'evaluator': evaluator,
        'device': config.device,
        'eval_mb_size': config.batch_size * 2  # Larger batch for evaluation
    }
    
    # Strategy Selection
    if strategy_name == "naive":
        print("→ Strategy: Naive Fine-tuning (Baseline)")
        print("   Expected: High catastrophic forgetting")
        return Naive(model, optimizer, criterion, **base_kwargs)
    
    elif strategy_name == "ewc":
        print(f"→ Strategy: Elastic Weight Consolidation (λ={config.ewc_lambda})")
        print("   Expected: Reduced forgetting via weight regularization")
        return EWC(
            model, optimizer, criterion, 
            ewc_lambda=config.ewc_lambda,
            **base_kwargs
        )
    
    elif strategy_name == "replay_balanced":
        print(f"→ Strategy: Class-Balanced Experience Replay (mem={config.memory_size})")
        print("   Expected: Balanced performance across all tasks")
        
        # Class-balanced storage ensures fair representation
        storage_policy = ClassBalancedBuffer(
            max_size=config.memory_size,
            adaptive_size=True  # Adapt as number of classes grows
        )
        
        replay = ReplayPlugin(
            mem_size=config.memory_size,
            storage_policy=storage_policy
        )
        
        return Naive(
            model, optimizer, criterion, 
            plugins=[replay], 
            **base_kwargs
        )
    
    elif strategy_name == "clear_replay":
        print(f"→ Strategy: CLEAR (Replay + Stored-Logits + Distillation, mem={config.memory_size})")
        storage_policy = ClassBalancedBuffer(
            max_size=config.memory_size,
            adaptive_size=True
        )

        # Use our CLEAR plugin which extends replay with stored logits + distillation
        clear = ClearReplayPlugin(
            mem_size=config.memory_size,
            storage_policy=storage_policy,
            clear_lambda=getattr(config, 'clear_lambda', 1.0),
            temperature=getattr(config, 'clear_temperature', 2.0)
        )

        return Naive(
            model, optimizer, criterion,
            plugins=[clear],
            **base_kwargs
        )
    
    elif strategy_name == "lora":
        print("→ Strategy: LoRA Fine-tuning")
        print("   Expected: Parameter-efficient learning")
        return Naive(model, optimizer, criterion, **base_kwargs)
    
    elif strategy_name == "hybrid_lora":
        print("→ Strategy: Hybrid (LoRA + Replay + Distillation)")
        print("   Expected: Best overall performance")
        
        from models import HybridCLStrategy
        
        storage_policy = ClassBalancedBuffer(
            max_size=config.memory_size,
            adaptive_size=True
        )
        
        replay = ReplayPlugin(
            mem_size=config.memory_size,
            storage_policy=storage_policy
        )
        
        return HybridCLStrategy(
            model, optimizer, criterion, config,
            plugins=[replay],
            **base_kwargs
        )
    
    elif strategy_name == "frozen_backbone":
        print("→ Strategy: Frozen Backbone (Linear Probing)")
        print("   Expected: Fast training, moderate forgetting")
        return Naive(model, optimizer, criterion, **base_kwargs)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


class AdvancedMetrics:
    """
    Advanced metrics for continual learning analysis.
    
    Goes beyond standard accuracy to measure:
    - Stability vs. Plasticity trade-off
    - Forward transfer to unseen tasks
    - Per-class forgetting patterns
    """
    
    def __init__(self):
        self.task_accuracies = []
        self.forgetting_events = []
        
    def compute_stability_plasticity(self, accuracies):
        """
        Compute stability-plasticity trade-off.
        
        Stability: Ability to retain old knowledge (resist forgetting)
        Plasticity: Ability to learn new tasks
        
        Args:
            accuracies: List of accuracies after each task
            
        Returns:
            (stability_score, plasticity_score) tuple
        """
        if len(accuracies) < 2:
            return 0.0, 0.0
        
        # Stability: retention rate on old tasks
        stability_scores = []
        for task_idx in range(len(accuracies) - 1):
            initial_acc = accuracies[task_idx]
            final_acc = accuracies[-1]
            retention = final_acc / (initial_acc + 1e-8)
            stability_scores.append(retention)
        
        stability = np.mean(stability_scores) if stability_scores else 0.0
        
        # Plasticity: performance on newest task (normalized)
        plasticity = accuracies[-1] / 100.0
        
        return stability, plasticity
    
    def compute_forward_transfer(self, first_task_acc, final_task_acc):
        """
        Measure positive forward transfer.
        
        Forward transfer: when learning task 1 helps with task 5
        
        Args:
            first_task_acc: Accuracy on first task
            final_task_acc: Accuracy on final task
            
        Returns:
            Forward transfer score
        """
        # Compare to random baseline (20% for 5-way classification)
        baseline = 20.0
        return (final_task_acc - baseline) / (100.0 - baseline)