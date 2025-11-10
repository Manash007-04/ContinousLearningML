# ===================================================================
# Continual Learning with Pre-Trained Models (ViT)
#
# This script demonstrates and compares several Continual Learning (CL)
# strategies on the SplitCIFAR-100 benchmark, using a pre-trained
# Vision Transformer (ViT-B/16).
#
# The strategies are chosen based on recent research, including:
# 1. paper1.pdf: "Continual Learning with Pre-Trained Models: A Survey"
# 2. paper2.pdf: "Experience Replay for Continual Learning" (CLEAR)
# 3. paper3.pdf: "Adaptive Memory Replay for Continual Learning"
#
# Author: [Your Name]
# ===================================================================

import torch
import torch.nn as nn
from torch.optim import SGD
from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.training.strategies import Naive, EWC
from avalanche.training.plugins import ReplayPlugin, EvaluationPlugin
from avalanche.training.storage import ClassBalancedBuffer
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics, bwt_metrics
from avalanche.logging import InteractiveLogger
from timm import create_model
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------------------
# 1. SETUP
# --------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Split CIFAR-100 into 5 tasks (experiences) of 20 classes each
benchmark = SplitCIFAR100(n_experiences=5, return_task_id=False, seed=42)

# Evaluation plugin to track metrics
interactive_logger = InteractiveLogger()
eval_plugin = EvaluationPlugin(
    accuracy_metrics(epoch=True, experience=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    bwt_metrics(experience=True, stream=True),
    loggers=[interactive_logger]
)

# --------------------------------------------------------------
# 2. MODEL DEFINITION (Pretrained Vision Transformer)
# --------------------------------------------------------------

def get_model(device, frozen_backbone=False):
    """
    Creates a fresh ViT-B/16 model, pre-trained on ImageNet-21k.
    Each strategy MUST get its own model instance.
    """
    model = create_model('vit_base_patch16_224', pretrained=True)

    if frozen_backbone:
        # --- This is the "Representation-based Method" ---
        # Inspired by paper1.pdf, Sec 3.2. Freeze the powerful
        # PTM backbone and only train the classifier head.
        print("Model: ViT-B/16 (Frozen Backbone)")
        for param in model.parameters():
            param.requires_grad = False
    else:
        print("Model: ViT-B/16 (Full Fine-tuning)")
        
    # Replace the head for 100 classes
    # New layers have requires_grad=True by default
    model.head = nn.Linear(model.head.in_features, 100)
    
    return model.to(device)

# --------------------------------------------------------------
# 3. STRATEGY CONFIGURATION
# --------------------------------------------------------------

def get_strategy(strategy_name, model, device, evaluator, lr=0.001, ewc_lambda=1000.0, mem_size=1000):
    """
    Returns the appropriate Avalanche strategy object.
    """
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # --- Base number of epochs for each task ---
    # 1 epoch is too low. 3-5 is better for a demo.
    # For a "real" run, this would be 10-20+.
    TRAIN_EPOCHS = 3 

    if strategy_name == "naive":
        # --- STRATEGY 1: Naive (Fine-tuning) ---
        # This is the "Sequential (no CLEAR)" baseline [cite: 206] from
        # paper2.pdf. It shows maximum catastrophic forgetting.
        print("Strategy: Naive (Baseline)")
        strategy = Naive(model, optimizer, criterion, train_mb_size=32,
                         train_epochs=TRAIN_EPOCHS, evaluator=evaluator, device=device)
    
    elif strategy_name == "ewc":
        # --- STRATEGY 2: EWC (Regularization) ---
        # A classic regularization method. Used as a comparison in
        # paper2.pdf[cite: 399].
        # Lambda must be high (e.g., 100-10000) to be effective.
        print(f"Strategy: EWC (lambda={ewc_lambda})")
        strategy = EWC(model, optimizer, criterion, ewc_lambda=ewc_lambda,
                       train_mb_size=32, train_epochs=TRAIN_EPOCHS,
                       evaluator=evaluator, device=device)
    
    elif strategy_name == "replay_uniform":
        # --- STRATEGY 3: Replay (Uniform Sampling) ---
        # This is the "off-policy replay alone"  from
        # paper2.pdf (i.e., "CLEAR without behavioral cloning").
        # It uses a standard reservoir buffer[cite: 347].
        print(f"Strategy: Replay (Uniform, mem_size={mem_size})")
        replay_plugin = ReplayPlugin(
            mem_size=mem_size
        )
        strategy = Naive(model, optimizer, criterion, train_mb_size=32,
                         train_epochs=TRAIN_EPOCHS, evaluator=evaluator,
                         plugins=[replay_plugin], device=device)
                         
    elif strategy_name == "replay_class_balanced":
        # --- STRATEGY 4: Replay (Class-Balanced Sampling) ---
        # Inspired by paper3.pdf, which argues for *adaptive*,
        # non-uniform replay over simple random sampling.
        # This uses a buffer that ensures all classes are
        # represented, preventing bias towards recent tasks.
        print(f"Strategy: Replay (Class-Balanced, mem_size={mem_size})")
        storage_policy = ClassBalancedBuffer(
            max_size=mem_size,
            adaptive_size=True  # Shrinks buffer space per class as tasks increase
        )
        replay_plugin = ReplayPlugin(
            mem_size=mem_size,
            storage_policy=storage_policy
        )
        strategy = Naive(model, optimizer, criterion, train_mb_size=32,
                         train_epochs=TRAIN_EPOCHS, evaluator=evaluator,
                         plugins=[replay_plugin], device=device)
                         
    elif strategy_name == "frozen_backbone":
        # --- STRATEGY 5: Naive (Frozen Backbone) ---
        # This implements the "Representation-based" PTM-CL method
        # from paper1.pdf. We use the Naive strategy
        # *on a model where the backbone is frozen* (done in get_model).
        print("Strategy: Naive (Frozen Backbone)")
        # Note: The optimizer will only get gradients from the unfrozen head.
        strategy = Naive(model, optimizer, criterion, train_mb_size=32,
                         train_epochs=TRAIN_EPOCHS, evaluator=evaluator, device=device)
    else:
        raise ValueError("Unknown strategy name")
        
    return strategy

# --------------------------------------------------------------
# 4. TRAINING LOOP
# --------------------------------------------------------------

strategies_to_run = [
    "naive",
    "ewc",
    "replay_uniform",
    "replay_class_balanced",
    "frozen_backbone"
]

results_log = []

for strategy_name in strategies_to_run:
    print(f"\n========================================================")
    print(f"===== RUNNING STRATEGY: {strategy_name.upper()} =====")
    print(f"========================================================")
    
    # 1. Get a fresh model for this strategy
    is_frozen = (strategy_name == "frozen_backbone")
    model_instance = get_model(device, frozen_backbone=is_frozen)
    
    # 2. Get the strategy
    cl_strategy = get_strategy(strategy_name, model_instance, device, eval_plugin)
    
    # 3. Run the training loop
    for experience in benchmark.train_stream:
        print(f"\n--- Training on Task {experience.current_experience + 1} ---")
        cl_strategy.train(experience)
        
        # 4. Evaluate on the *entire* test stream
        print(f"--- Evaluating on all {experience.current_experience + 1} tasks ---")
        eval_results = cl_strategy.eval(benchmark.test_stream)
        
        # Log results
        acc = eval_results['Top1_Acc_Stream/eval_phase/test_stream']
        results_log.append({
            "Strategy": strategy_name,
            "Task": experience.current_experience + 1,
            "Accuracy": acc
        })
        print(f"Task {experience.current_experience + 1} Avg Accuracy: {acc:.2f}%")

# --------------------------------------------------------------
# 5. SAVE AND VISUALIZE RESULTS
# --------------------------------------------------------------

# Convert log to DataFrame
df = pd.DataFrame(results_log)
df.to_csv("cl_vit_results.csv", index=False)
print("\nResults saved to cl_vit_results.csv")

# Create a professional-looking plot with Seaborn
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

g = sns.lineplot(
    data=df,
    x='Task',
    y='Accuracy',
    hue='Strategy',
    style='Strategy',
    markers=True,
    dashes=False,
    linewidth=2.5
)

g.set_title("Continual Learning Performance on SplitCIFAR-100 (ViT-B/16)", fontsize=16)
g.set_ylabel("Average Stream Accuracy (%)", fontsize=12)
g.set_xlabel("Tasks Completed", fontsize=12)
g.set(xticks=list(range(1, 6)), yticks=list(range(0, 101, 10)))
g.set_ylim(0, 100) # Ensure y-axis is 0-100

plt.legend(title='CL Strategy', loc='best', borderaxespad=0., fontsize=11)
plt.tight_layout()
plt.savefig("cl_vit_performance.png", dpi=300)
plt.show()

print("\nVisualization saved to cl_vit_performance.png")
print("Experiment complete.")# ===================================================================
# Continual Learning with Pre-Trained Models (ViT)
#
# This script demonstrates and compares several Continual Learning (CL)
# strategies on the SplitCIFAR-100 benchmark, using a pre-trained
# Vision Transformer (ViT-B/16).
#
# The strategies are chosen based on recent research, including:
# 1. paper1.pdf: "Continual Learning with Pre-Trained Models: A Survey"
# 2. paper2.pdf: "Experience Replay for Continual Learning" (CLEAR)
# 3. paper3.pdf: "Adaptive Memory Replay for Continual Learning"
#
# Author: [Your Name]
# ===================================================================

import torch
import torch.nn as nn
from torch.optim import SGD
from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.training.strategies import Naive, EWC
from avalanche.training.plugins import ReplayPlugin, EvaluationPlugin
from avalanche.training.storage import ClassBalancedBuffer
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics, bwt_metrics
from avalanche.logging import InteractiveLogger
from timm import create_model
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------------------
# 1. SETUP
# --------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Split CIFAR-100 into 5 tasks (experiences) of 20 classes each
benchmark = SplitCIFAR100(n_experiences=5, return_task_id=False, seed=42)

# Evaluation plugin to track metrics
interactive_logger = InteractiveLogger()
eval_plugin = EvaluationPlugin(
    accuracy_metrics(epoch=True, experience=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    bwt_metrics(experience=True, stream=True),
    loggers=[interactive_logger]
)

# --------------------------------------------------------------
# 2. MODEL DEFINITION (Pretrained Vision Transformer)
# --------------------------------------------------------------

def get_model(device, frozen_backbone=False):
    """
    Creates a fresh ViT-B/16 model, pre-trained on ImageNet-21k.
    Each strategy MUST get its own model instance.
    """
    model = create_model('vit_base_patch16_224', pretrained=True)

    if frozen_backbone:
        # --- This is the "Representation-based Method" ---
        # Inspired by paper1.pdf, Sec 3.2. Freeze the powerful
        # PTM backbone and only train the classifier head.
        print("Model: ViT-B/16 (Frozen Backbone)")
        for param in model.parameters():
            param.requires_grad = False
    else:
        print("Model: ViT-B/16 (Full Fine-tuning)")
        
    # Replace the head for 100 classes
    # New layers have requires_grad=True by default
    model.head = nn.Linear(model.head.in_features, 100)
    
    return model.to(device)

# --------------------------------------------------------------
# 3. STRATEGY CONFIGURATION
# --------------------------------------------------------------

def get_strategy(strategy_name, model, device, evaluator, lr=0.001, ewc_lambda=1000.0, mem_size=1000):
    """
    Returns the appropriate Avalanche strategy object.
    """
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # --- Base number of epochs for each task ---
    # 1 epoch is too low. 3-5 is better for a demo.
    # For a "real" run, this would be 10-20+.
    TRAIN_EPOCHS = 3 

    if strategy_name == "naive":
        # --- STRATEGY 1: Naive (Fine-tuning) ---
        # This is the "Sequential (no CLEAR)" baseline [cite: 206] from
        # paper2.pdf. It shows maximum catastrophic forgetting.
        print("Strategy: Naive (Baseline)")
        strategy = Naive(model, optimizer, criterion, train_mb_size=32,
                         train_epochs=TRAIN_EPOCHS, evaluator=evaluator, device=device)
    
    elif strategy_name == "ewc":
        # --- STRATEGY 2: EWC (Regularization) ---
        # A classic regularization method. Used as a comparison in
        # paper2.pdf[cite: 399].
        # Lambda must be high (e.g., 100-10000) to be effective.
        print(f"Strategy: EWC (lambda={ewc_lambda})")
        strategy = EWC(model, optimizer, criterion, ewc_lambda=ewc_lambda,
                       train_mb_size=32, train_epochs=TRAIN_EPOCHS,
                       evaluator=evaluator, device=device)
    
    elif strategy_name == "replay_uniform":
        # --- STRATEGY 3: Replay (Uniform Sampling) ---
        # This is the "off-policy replay alone"  from
        # paper2.pdf (i.e., "CLEAR without behavioral cloning").
        # It uses a standard reservoir buffer[cite: 347].
        print(f"Strategy: Replay (Uniform, mem_size={mem_size})")
        replay_plugin = ReplayPlugin(
            mem_size=mem_size
        )
        strategy = Naive(model, optimizer, criterion, train_mb_size=32,
                         train_epochs=TRAIN_EPOCHS, evaluator=evaluator,
                         plugins=[replay_plugin], device=device)
                         
    elif strategy_name == "replay_class_balanced":
        # --- STRATEGY 4: Replay (Class-Balanced Sampling) ---
        # Inspired by paper3.pdf, which argues for *adaptive*,
        # non-uniform replay over simple random sampling.
        # This uses a buffer that ensures all classes are
        # represented, preventing bias towards recent tasks.
        print(f"Strategy: Replay (Class-Balanced, mem_size={mem_size})")
        storage_policy = ClassBalancedBuffer(
            max_size=mem_size,
            adaptive_size=True  # Shrinks buffer space per class as tasks increase
        )
        replay_plugin = ReplayPlugin(
            mem_size=mem_size,
            storage_policy=storage_policy
        )
        strategy = Naive(model, optimizer, criterion, train_mb_size=32,
                         train_epochs=TRAIN_EPOCHS, evaluator=evaluator,
                         plugins=[replay_plugin], device=device)
                         
    elif strategy_name == "frozen_backbone":
        # --- STRATEGY 5: Naive (Frozen Backbone) ---
        # This implements the "Representation-based" PTM-CL method
        # from paper1.pdf. We use the Naive strategy
        # *on a model where the backbone is frozen* (done in get_model).
        print("Strategy: Naive (Frozen Backbone)")
        # Note: The optimizer will only get gradients from the unfrozen head.
        strategy = Naive(model, optimizer, criterion, train_mb_size=32,
                         train_epochs=TRAIN_EPOCHS, evaluator=evaluator, device=device)
    else:
        raise ValueError("Unknown strategy name")
        
    return strategy

# --------------------------------------------------------------
# 4. TRAINING LOOP
# --------------------------------------------------------------

strategies_to_run = [
    "naive",
    "ewc",
    "replay_uniform",
    "replay_class_balanced",
    "frozen_backbone"
]

results_log = []

for strategy_name in strategies_to_run:
    print(f"\n========================================================")
    print(f"===== RUNNING STRATEGY: {strategy_name.upper()} =====")
    print(f"========================================================")
    
    # 1. Get a fresh model for this strategy
    is_frozen = (strategy_name == "frozen_backbone")
    model_instance = get_model(device, frozen_backbone=is_frozen)
    
    # 2. Get the strategy
    cl_strategy = get_strategy(strategy_name, model_instance, device, eval_plugin)
    
    # 3. Run the training loop
    for experience in benchmark.train_stream:
        print(f"\n--- Training on Task {experience.current_experience + 1} ---")
        cl_strategy.train(experience)
        
        # 4. Evaluate on the *entire* test stream
        print(f"--- Evaluating on all {experience.current_experience + 1} tasks ---")
        eval_results = cl_strategy.eval(benchmark.test_stream)
        
        # Log results
        acc = eval_results['Top1_Acc_Stream/eval_phase/test_stream']
        results_log.append({
            "Strategy": strategy_name,
            "Task": experience.current_experience + 1,
            "Accuracy": acc
        })
        print(f"Task {experience.current_experience + 1} Avg Accuracy: {acc:.2f}%")

# --------------------------------------------------------------
# 5. SAVE AND VISUALIZE RESULTS
# --------------------------------------------------------------

# Convert log to DataFrame
df = pd.DataFrame(results_log)
df.to_csv("cl_vit_results.csv", index=False)
print("\nResults saved to cl_vit_results.csv")

# Create a professional-looking plot with Seaborn
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

g = sns.lineplot(
    data=df,
    x='Task',
    y='Accuracy',
    hue='Strategy',
    style='Strategy',
    markers=True,
    dashes=False,
    linewidth=2.5
)

g.set_title("Continual Learning Performance on SplitCIFAR-100 (ViT-B/16)", fontsize=16)
g.set_ylabel("Average Stream Accuracy (%)", fontsize=12)
g.set_xlabel("Tasks Completed", fontsize=12)
g.set(xticks=list(range(1, 6)), yticks=list(range(0, 101, 10)))
g.set_ylim(0, 100) # Ensure y-axis is 0-100

plt.legend(title='CL Strategy', loc='best', borderaxespad=0., fontsize=11)
plt.tight_layout()
plt.savefig("cl_vit_performance.png", dpi=300)
plt.show()

print("\nVisualization saved to cl_vit_performance.png")
print("Experiment complete.")