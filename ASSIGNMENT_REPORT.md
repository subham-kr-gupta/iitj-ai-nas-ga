# Neural Architecture Search with Genetic Algorithm - Assignment Report

---

## Code & Logs Repository

GitHub Link: https://github.com/subham-kr-gupta/iitj-ai-nas-ga

The repository contains:
- Modified code (model_ga.py with Q1A & Q2B implementations)
- Complete outputs folder with logs (outputs/run_1/)
- All supporting files (model_cnn.py, nas_run.py)

Note: The data/ folder is not included as it contains the publicly 
available CIFAR-10 dataset (170MB). The code will automatically 
download CIFAR-10 when run.

---

## Modifications Made

### Q1A: Roulette-Wheel Selection

#### Modified Code Section
**File**: `model_ga.py`

```python
def selection(self):
    """Roulette-Wheel selection based on relative fitness scores"""
    selected = []
    
    # Calculate relative fitness scores
    fitness_values = [arch.fitness for arch in self.population]
    min_fitness = min(fitness_values)
    
    # Shift fitness values to make them all positive (if needed)
    if min_fitness < 0:
        shifted_fitness = [f - min_fitness + 0.001 for f in fitness_values]
    else:
        shifted_fitness = [f + 0.001 for f in fitness_values]  # Add small epsilon to avoid zero division
    
    # Calculate total fitness
    total_fitness = sum(shifted_fitness)
    
    # Calculate selection probabilities proportional to fitness
    probabilities = [f / total_fitness for f in shifted_fitness]
    
    # Log the relative fitness and probabilities for this generation
    print(f"\n--- Roulette-Wheel Selection ---", flush=True)
    for i, arch in enumerate(self.population):
        print(f"  Arch {i+1}: Fitness={arch.fitness:.6f}, Relative Fitness={shifted_fitness[i]:.6f}, Probability={probabilities[i]:.4f}", flush=True)
    
    # Perform selection using roulette wheel (random.choices with weights)
    for _ in range(self.population_size):
        selected_arch = random.choices(self.population, weights=probabilities, k=1)[0]
        selected.append(selected_arch)
    
    return selected
```

#### Justification

**1. Mathematical Foundation**

The Roulette-Wheel selection (also known as Fitness Proportionate Selection) assigns selection probability to each chromosome proportional to its fitness score. For a population of size $N$, the selection probability for chromosome $i$ is:

$$P_i = \frac{f_i}{\sum_{j=1}^{N} f_j}$$

where $f_i$ is the fitness score of chromosome $i$.

**2. Implementation Details**

- **Fitness Normalization**: Since fitness scores can be negative or very small, we apply a shifting transformation:
  - If $\min(f_i) < 0$: $f_i' = f_i - \min(f_j) + \epsilon$
  - Otherwise: $f_i' = f_i + \epsilon$
  
  where $\epsilon = 0.001$ is a small constant to prevent zero division.

- **Probability Calculation**: After normalization, probabilities are computed as:
  $$P_i = \frac{f_i'}{\sum_{j=1}^{N} f_j'}$$

- **Selection Mechanism**: Python's `random.choices()` function implements weighted random selection, which is equivalent to spinning a roulette wheel where each chromosome's "slice" is proportional to its fitness.

**3. Advantages over Tournament Selection**

- **Proportional Representation**: Highly fit individuals have proportionally higher selection chances
- **Diversity Preservation**: Even lower-fitness individuals have non-zero selection probability
- **Transparency**: Selection probabilities are explicitly computed and logged

**4. Logging Enhancement**

The modified code logs:
- Original fitness scores
- Normalized relative fitness scores
- Computed selection probabilities

This provides complete transparency into the selection process for analysis.

---

### Q2B: Modified Fitness Function with Separate Conv and FC Penalties

#### Modified Code Section
**File**: `model_ga.py`  

```python
# Calculate model complexity penalty separately for Conv and FC layers
conv_params = 0
fc_params = 0

for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        conv_params += sum(p.numel() for p in module.parameters())
    elif isinstance(module, nn.Linear):
        fc_params += sum(p.numel() for p in module.parameters())

# Weight factors for penalty calculation
# Conv layers: Higher computational cost due to spatial operations (O(C_in × C_out × K² × H × W))
# FC layers: Lower computational cost (O(N_in × N_out))
# Conv operations are ~2-3x more expensive per parameter, so we assign higher weight
weight_conv = 2.5  # Higher weight for conv parameters
weight_fc = 1.0    # Base weight for FC parameters

# Normalize parameters (per million) and apply weights
conv_penalty = (conv_params / 1e6) * weight_conv
fc_penalty = (fc_params / 1e6) * weight_fc
complexity_penalty = conv_penalty + fc_penalty

# Log parameter counts and penalties
print(f"\n  Parameter Analysis:", flush=True)
print(f"    Conv Params: {conv_params:,} | FC Params: {fc_params:,}", flush=True)
print(f"    Conv Penalty: {conv_penalty:.6f} | FC Penalty: {fc_penalty:.6f}", flush=True)
print(f"    Total Complexity Penalty: {complexity_penalty:.6f}", flush=True)

del model, inputs, outputs, labels
torch.cuda.empty_cache()

# Fitness = accuracy - lambda * complexity
architecture.accuracy = best_acc
architecture.best_epoch = best_epoch
architecture.fitness = best_acc - 0.01 * complexity_penalty
print(f"    Final Fitness Score: {architecture.fitness:.6f}", flush=True)
```

#### Justification

**1. Computational Complexity Analysis**

The computational cost of neural network layers differs significantly:

**Convolutional Layers:**
- Forward pass complexity: $O(C_{in} \times C_{out} \times K^2 \times H \times W)$
- Where:
  - $C_{in}$, $C_{out}$ = input/output channels
  - $K$ = kernel size
  - $H \times W$ = spatial dimensions of feature map
- Number of parameters: $C_{in} \times C_{out} \times K^2 + C_{out}$ (weights + bias)

**Fully Connected (FC) Layers:**
- Forward pass complexity: $O(N_{in} \times N_{out})$
- Where:
  - $N_{in}$ = input dimension
  - $N_{out}$ = output dimension
- Number of parameters: $N_{in} \times N_{out} + N_{out}$ (weights + bias)

**2. Weight Selection Rationale**

The key difference is that convolutions operate on 2D spatial data with multiple applications of the same kernel across the spatial dimensions. This makes convolutions computationally more expensive per parameter.

**Empirical Analysis:**
- Consider a Conv layer: 64 input channels, 128 output channels, 3×3 kernel, operating on 32×32 feature maps
  - Parameters: $64 \times 128 \times 3 \times 3 = 73,728$
  - FLOPs: $64 \times 128 \times 9 \times 32 \times 32 \approx 75.5M$
  - FLOPs per parameter: $\frac{75.5M}{73,728} \approx 1,024$

- Consider an FC layer: 1024 input, 512 output
  - Parameters: $1024 \times 512 = 524,288$
  - FLOPs: $1024 \times 512 \approx 0.52M$
  - FLOPs per parameter: $\frac{0.52M}{524,288} \approx 1$

This shows that **Conv operations require ~1000× more FLOPs per parameter** than FC operations when spatial dimensions are considered.

**3. Weight Assignment**

Given the computational disparity, we assign:
- $w_{conv} = 2.5$: Higher weight for convolutional parameters
- $w_{fc} = 1.0$: Base weight for fully connected parameters

**Fitness Function:**
$$\text{Fitness} = \text{Accuracy} - \lambda \times \left(\frac{P_{conv}}{10^6} \times w_{conv} + \frac{P_{fc}}{10^6} \times w_{fc}\right)$$

where:
- $P_{conv}$ = total convolutional parameters
- $P_{fc}$ = total fully connected parameters
- $\lambda = 0.01$ = penalty coefficient

**4. Justification for Weight Ratio (2.5:1)**

While theoretical analysis suggests Conv layers are much more expensive, we choose a moderate ratio of 2.5:1 because:

1. **Hardware Optimization**: Modern GPUs are heavily optimized for convolution operations through specialized CUDA kernels
2. **Memory Access Patterns**: Convolutions have better cache locality compared to FC layers
3. **Practical Balance**: Too high a penalty would eliminate all but the smallest Conv architectures
4. **NAS Goal**: We want to balance model capacity (parameters) with computational efficiency, not just minimize FLOPs

---