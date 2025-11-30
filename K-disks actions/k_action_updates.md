# K-Action (K-Disks) Integration Documentation

This document provides a comprehensive guide to the K-action (K-disks) integration into the CtRL-Sim project. The K-action space is an alternative action discretization scheme that uses vocabulary-based action tokens instead of the original uniform grid discretization (accel × steer).

## Table of Contents
1. [Overview](#overview)
2. [Action Space Comparison](#action-space-comparison)
3. [Configuration Files](#configuration-files)
4. [Code Modifications](#code-modifications)
5. [Data Pipeline](#data-pipeline)
6. [Usage Guide](#usage-guide)
7. [Limitations and Future Work](#limitations-and-future-work)

---

## Overview

### Original Action Space (Grid)
The original CtRL-Sim uses a uniform grid discretization:
- `action_dim = accel_discretization × steer_discretization` (default: 20 × 50 = 1000)
- Actions are discretized uniformly across acceleration [-10, 10] and steering [-0.7, 0.7]

### New Action Space (K-Action)
The K-action space uses a pre-computed vocabulary:
- `action_dim = k_action_vocab_size` (default: 384)
- Actions are tokenized using K-disks clustering, providing better coverage of realistic driving behaviors
- Each timestep has a single integer token (0 to vocab_size-1)

---

## Action Space Comparison

| Aspect | Grid Action Space | K-Action Space |
|--------|------------------|----------------|
| Dimensionality | accel_disc × steer_disc (1000) | vocab_size (384) |
| Discretization | Uniform grid | K-disks clustering |
| Coverage | Uniform across range | Concentrated on realistic actions |
| Storage | 2D (accel, steer) → 1D index | 1D token directly |
| Source | Computed from continuous actions | Pre-stored in JSON (`k_action` field) |

---

## Configuration Files

### 1. `cfgs/dataset/waymo/base.yaml`

**Added fields:**
```yaml
action_space: grid  # options: grid, k_action
k_action_vocab_size: 384
k_action_vocab_path: '${project_root}/K-disks actions/k_disks_vocab_384_10Hz_seed42.pkl'
```

**Rationale:**
- `action_space`: Master toggle controlling which action space the entire pipeline uses
- `k_action_vocab_size`: Defines the output dimension when using k_action mode (must match vocabulary file)
- `k_action_vocab_path`: Path to the vocabulary file for k-disks encoding/decoding (used for future execution)

### 2. `cfgs/model/base.yaml`

**Added fields:**
```yaml
action_space: ${dataset.waymo.action_space}  # Inherits from dataset config
use_residual_mlp: True
residual_mlp_layers: 2
```

**Rationale:**
- `action_space`: Mirrors dataset setting to ensure consistency across the pipeline
- `use_residual_mlp`: Enables ResidualMLP layers (from Scenario Dreamer) for better feature extraction
- `residual_mlp_layers`: Number of hidden layers in ResidualMLP blocks

---

## Code Modifications

### 1. `utils/layers.py` - ResidualMLP Class

**Added:**
```python
class ResidualMLP(nn.Module):
    """
    Residual feed-forward block with configurable depth.
    
    Architecture:
        Input → [Linear → LayerNorm → ReLU] × (n_hidden-1) → Linear → LayerNorm
                ↓                                                    ↓
                Linear → LayerNorm ────────────────────────────────→ + → ReLU → [Linear → Output]
    """
    def __init__(self, input_dim, hidden_dim, n_hidden=2, output_dim=None):
        ...
```

**Rationale:**
- Inspired by Scenario Dreamer's architecture for more expressive feature transformations
- Residual connections help with gradient flow in deeper networks
- LayerNorm provides stable training
- Optional output projection for prediction heads

**Key features:**
- Main pathway: `n_hidden` Linear layers with LayerNorm (ReLU between layers)
- Residual pathway: Single Linear + LayerNorm for dimension matching
- Final: ReLU(main + residual), then optional output Linear

---

### 2. `modules/encoder.py` - Encoder Class

**Changes:**

#### a. Action space configuration (lines 15-23):
```python
self.use_k_actions = getattr(self.cfg_rl_waymo, 'action_space', 'grid') == 'k_action'
self.use_residual_mlp = getattr(self.cfg_model, 'use_residual_mlp', False)
self.residual_layers = getattr(self.cfg_model, 'residual_mlp_layers', 2)
self.grid_action_dim = self.cfg_rl_waymo.accel_discretization * self.cfg_rl_waymo.steer_discretization
self.k_action_dim = getattr(self.cfg_rl_waymo, 'k_action_vocab_size', None)
if self.use_k_actions and self.k_action_dim is None:
    raise ValueError("cfg.dataset.waymo.k_action_vocab_size must be set when action_space is k_action.")
self.action_dim = self.k_action_dim if self.use_k_actions else self.grid_action_dim
```

**Rationale:**
- Determines action embedding dimension based on chosen action space
- Validates configuration: k_action mode requires vocab_size to be set
- Stores both dimensions for potential dual-head scenarios

#### b. Embedding layers use ResidualMLP when enabled (lines 28-31):
```python
self.embed_state = self._build_mlp(self.cfg_model.state_dim, self.cfg_model.hidden_dim)
self.embed_goal = self._build_mlp(self.cfg_rl_waymo.goal_dim, self.cfg_model.hidden_dim)
self.embed_state_goal = self._build_mlp(self.cfg_model.hidden_dim * 2, self.cfg_model.hidden_dim)
self.embed_action = nn.Embedding(int(self.action_dim), self.cfg_model.hidden_dim)
```

**Rationale:**
- `_build_mlp` helper method returns either `ResidualMLP` or `MLPLayer` based on config
- Action embedding size matches the selected action space dimension
- Enables consistent architecture across all embedding pathways

#### c. `_build_mlp` helper method (lines 56-62):
```python
def _build_mlp(self, input_dim, output_dim):
    if self.use_residual_mlp:
        return ResidualMLP(input_dim=input_dim,
                           hidden_dim=self.cfg_model.hidden_dim,
                           n_hidden=self.residual_layers,
                           output_dim=output_dim)
    return MLPLayer(input_dim, self.cfg_model.hidden_dim, output_dim)
```

**Rationale:**
- Centralizes MLP construction logic
- Allows easy switching between original MLPLayer and new ResidualMLP
- Maintains backward compatibility when `use_residual_mlp=False`

#### d. Action selection in forward pass (lines 72-78):
```python
if self.use_k_actions:
    if not hasattr(data['agent'], 'k_actions'):
        raise ValueError("action_space is set to k_action but 'k_actions' not found in data.")
    actions = data['agent'].k_actions
else:
    actions = data['agent'].actions
```

**Rationale:**
- Selects correct action tensor based on configuration
- Provides clear error message if k_actions missing when expected
- Ensures data consistency throughout training/inference

---

### 3. `modules/decoder.py` - Decoder Class

**Changes:**

#### a. Dual action prediction heads (lines 27-28):
```python
self.predict_grid_action = self._build_mlp(self.cfg_model.hidden_dim, self.grid_action_dim)
self.predict_k_action = self._build_mlp(self.cfg_model.hidden_dim, self.k_action_dim) if self.k_action_dim is not None else None
```

**Rationale:**
- **Always** builds grid action head (maintains backward compatibility)
- **Conditionally** builds k_action head only when vocab_size is configured
- Both heads can coexist, enabling future multi-task learning or head switching

#### b. RTG and state prediction heads use ResidualMLP:
```python
if self.cfg_model.predict_rtg:
    self.predict_rtg = self._build_mlp(self.cfg_model.hidden_dim, ...)

if self.cfg_model.predict_future_states:
    self.predict_future_states = self._build_mlp(self.cfg_model.hidden_dim, ...)
```

**Rationale:**
- Consistent architecture across all prediction heads
- ResidualMLP provides stronger feature transformations for auxiliary tasks

#### c. `_reshape_actions` helper method (lines 45-48):
```python
def _reshape_actions(self, logits, batch_size, seq_len, action_dim):
    if logits is None:
        return None
    return logits.reshape(batch_size, seq_len, self.cfg_rl_waymo.max_num_agents, action_dim).permute(0, 2, 1, 3)
```

**Rationale:**
- Centralizes action tensor reshaping logic
- Handles None case gracefully (for conditional heads)
- Output shape: `[batch_size, num_agents, seq_len, action_dim]`

#### d. Forward pass with dual heads (lines 75-93):
```python
grid_action_logits = self.predict_grid_action(action_token) if self.predict_grid_action is not None else None
k_action_logits = self.predict_k_action(action_token) if self.predict_k_action is not None else None
grid_action_preds = self._reshape_actions(grid_action_logits, batch_size, seq_len, self.grid_action_dim)
k_action_preds = self._reshape_actions(k_action_logits, batch_size, seq_len, self.k_action_dim) if self.predict_k_action is not None else None

if grid_action_preds is not None:
    preds['grid_action_preds'] = grid_action_preds
if k_action_preds is not None:
    preds['k_action_preds'] = k_action_preds

if self.use_k_actions:
    if k_action_preds is None:
        raise ValueError("k_action head is not initialized but action_space is k_action.")
    preds['action_preds'] = k_action_preds
else:
    preds['action_preds'] = grid_action_preds
```

**Rationale:**
- Computes both action heads when available
- Stores both predictions in output dictionary (enables analysis/debugging)
- `preds['action_preds']` always contains the "active" prediction based on config
- Validates k_action head exists when needed

---

### 4. `models/ctrl_sim.py` - CtRLSim Model

**Changes:**

#### a. Action dimension configuration (lines 28-36):
```python
self.use_k_actions = getattr(self.cfg_rl_waymo, 'action_space', 'grid') == 'k_action'
self.grid_action_dim = self.cfg_rl_waymo.accel_discretization * self.cfg_rl_waymo.steer_discretization
self.k_action_dim = getattr(self.cfg_rl_waymo, 'k_action_vocab_size', None)
if self.use_k_actions and self.k_action_dim is None:
    raise ValueError("cfg.dataset.waymo.k_action_vocab_size must be set when action_space is k_action.")
self.action_dim = self.k_action_dim if self.use_k_actions else self.grid_action_dim
```

**Rationale:**
- Model-level configuration mirroring encoder/decoder
- Ensures `self.action_dim` used in loss computation matches predictions

#### b. Loss computation with k_actions (in `compute_loss`, both trajeglish and standard branches):
```python
if self.use_k_actions:
    if not hasattr(data['agent'], 'k_actions'):
        raise ValueError("k_action targets missing in batch while action_space is k_action.")
    actions = data['agent'].k_actions[:, :, 1:].reshape(-1)  # for trajeglish
    # OR
    actions = data['agent'].k_actions.view(-1)  # for standard
else:
    actions = data['agent'].actions[:, :, 1:].reshape(-1)  # for trajeglish
    # OR
    actions = data['agent'].actions.view(-1)  # for standard
```

**Rationale:**
- Selects correct action targets for cross-entropy loss
- Validates k_actions presence before use
- Maintains same loss computation logic, only changes target tensor

---

### 5. `datasets/rl_waymo/dataset.py` - Base Dataset

**Changes:**

#### `select_relevant_agents` method - added k_actions parameter:
```python
def select_relevant_agents(self, agent_states, agent_types, actions, rtgs, goals, 
                           origin_agent_idx, timestep, moving_agent_mask, 
                           relevant_agent_idxs=None, k_actions=None):
    ...
    final_k_actions = None
    if k_actions is not None:
        final_k_actions = np.zeros((self.cfg_dataset.max_num_agents, *k_actions[0].shape))
    ...
    if k_actions is not None:
        final_k_actions[:len(closest_ag_ids)] = k_actions[closest_ag_ids]
    ...
    # Updated return statements to include k_actions when present
```

**Rationale:**
- Extends agent selection to include k_actions tensor
- Maintains backward compatibility (k_actions is optional parameter)
- Filters and pads k_actions identically to other agent data

---

### 6. `datasets/rl_waymo/dataset_ctrl_sim.py` - CtRL-Sim Dataset

**Changes:**

#### a. `_get_raw_file_name` helper (lines 25-26):
```python
def _get_raw_file_name(self, idx):
    return os.path.splitext(os.path.basename(self.files[idx]))[0]
```

**Rationale:**
- Extracts base filename without extension for consistent file lookups
- Used when loading k_actions from raw JSON files

#### b. `_extract_k_actions` method (lines 29-38):
```python
def _extract_k_actions(self, agents_data):
    k_actions = []
    for agent in agents_data:
        if 'k_action' not in agent:
            return None
        k_actions.append(agent['k_action'])
    if len(k_actions) == 0:
        return None
    return np.array(k_actions, dtype=int)
```

**Rationale:**
- Extracts k_action arrays from raw JSON agent data
- Returns None if any agent lacks k_action field (graceful degradation)
- Converts to numpy array with integer dtype

#### c. `_load_k_actions_from_raw` method (lines 41-50):
```python
def _load_k_actions_from_raw(self, raw_file_name):
    raw_json_path = os.path.join(self.cfg_dataset.dataset_path, f"{self.split_name}", f"{raw_file_name}.json")
    if not os.path.exists(raw_json_path):
        return None
    with open(raw_json_path, 'r') as f:
        raw_data = json.load(f)
    if 'objects' not in raw_data:
        return None
    return self._extract_k_actions(raw_data['objects'])
```

**Rationale:**
- Fallback method to load k_actions from original JSON when not in pickle
- Useful for older preprocessed pickles that lack k_actions
- Handles missing file/data gracefully

#### d. Preprocessing path - saves k_actions to pickle:
```python
ag_k_actions = self._extract_k_actions(agent_data)
...
if ag_k_actions is not None:
    to_pickle['ag_k_actions'] = ag_k_actions
```

**Rationale:**
- Extracts k_actions during preprocessing
- Saves to pickle file for fast loading in future runs
- Conditional saving (only when k_actions available)

#### e. Loading path - loads k_actions from pickle or raw:
```python
ag_k_actions = data.get('ag_k_actions', None)
...
if ag_k_actions is None:
    ag_k_actions = self._load_k_actions_from_raw(raw_file_name)
if ag_k_actions is not None:
    ag_k_actions = np.array(ag_k_actions, dtype=int)
```

**Rationale:**
- First tries to load from preprocessed pickle
- Falls back to raw JSON if not found (backward compatibility)
- Ensures consistent dtype

#### f. Validation when k_action space required:
```python
if getattr(self.cfg_dataset, 'action_space', 'grid') == 'k_action' and k_actions is None:
    raise ValueError(f"k_action tokens not found for sample {raw_file_name} while action_space is k_action.")
```

**Rationale:**
- Fails fast with clear error message
- Prevents silent failures during training

#### g. Agent selection with k_actions:
```python
if k_actions is not None:
    (agent_states, agent_types, actions, rtgs, goals, moving_agent_mask,
     k_actions, new_origin_agent_idx) = self.select_relevant_agents(..., k_actions=k_actions)
else:
    agent_states, agent_types, actions, rtgs, goals, moving_agent_mask, new_origin_agent_idx = self.select_relevant_agents(...)
```

**Rationale:**
- Passes k_actions through agent selection/filtering pipeline
- Unpacks correctly based on presence

#### h. Data dictionary construction:
```python
agent_dict = {
    'agent_states': add_batch_dim(agent_states),
    'agent_types': add_batch_dim(agent_types), 
    'goals': add_batch_dim(goals),
    'actions': add_batch_dim(actions),
    'rtgs': add_batch_dim(rtgs),
    'timesteps': add_batch_dim(timesteps),
    'moving_agent_mask': add_batch_dim(moving_agent_mask)
}
if k_actions is not None:
    agent_dict['k_actions'] = add_batch_dim(k_actions)
d['agent'] = from_numpy(agent_dict)
```

**Rationale:**
- Restructured to use dictionary intermediate for cleaner conditional addition
- k_actions only added when available
- Maintains consistent tensor shapes with batch dimension

---

### 7. `policies/autoregressive_policy.py` - Policy Class

**Changes:**

```python
self.action_space = getattr(self.cfg_rl_waymo, 'action_space', 'grid')
if self.action_space == 'k_action':
    raise NotImplementedError("AutoregressivePolicy does not yet support k_action action_space.")
```

**Rationale:**
- Prevents using k_action mode with current policy implementation
- k_action requires additional implementation:
  - Loading k-disks vocabulary
  - Converting predicted tokens back to (accel, steer) pairs
  - Executing actions in simulator
- Clear error message guides future development

---

## Data Pipeline

### K-Action Data Format

In the raw JSON files (e.g., `offline_rl/val/*.json`):
```json
{
  "objects": [
    {
      "position": [...],
      "velocity": [...],
      "k_action": [156, 11, 156, 136, ...]  // 90 timesteps, integer tokens
    },
    ...
  ]
}
```

### Data Flow

```
Raw JSON (with k_action) 
    ↓ _extract_k_actions()
Preprocessing 
    ↓ save to pickle
Pickle (with ag_k_actions)
    ↓ load / _load_k_actions_from_raw() fallback
Filtering/Selection
    ↓ select_relevant_agents(k_actions=...)
Data Dictionary
    ↓ d['agent']['k_actions']
Model Training
    ↓ encoder: embed_action(k_actions)
    ↓ decoder: predict_k_action(...)
    ↓ loss: cross_entropy(k_action_preds, k_actions)
```

---

## Usage Guide

### Training with Grid Actions (Default)
No configuration changes needed:
```bash
python train.py
```

### Training with K-Actions
Modify config or use command line overrides:
```bash
python train.py dataset.waymo.action_space=k_action
```

Or edit `cfgs/dataset/waymo/base.yaml`:
```yaml
action_space: k_action
```

### Enabling ResidualMLP
```bash
python train.py model.use_residual_mlp=True model.residual_mlp_layers=2
```

### Preprocessing Data with K-Actions
Ensure JSON files have `k_action` field, then run preprocessing:
```bash
python train.py dataset.waymo.preprocess_real_data=True
```
This will create new pickle files with `ag_k_actions` field.

---

## Limitations and Future Work

### Current Limitations
1. **Policy Execution**: `AutoregressivePolicy` does not support k_action mode
   - Requires implementing k-disks vocabulary loading
   - Requires token-to-action conversion for simulator

2. **Vocabulary Dependency**: k_action mode requires pre-computed vocabulary file
   - Path configured in `k_action_vocab_path`
   - Must match `k_action_vocab_size`

3. **Data Requirement**: k_action mode requires `k_action` field in JSON files
   - Not all datasets may have this field
   - Fallback gracefully when `action_space=grid`

### Future Work
1. Implement k_action policy execution in `autoregressive_policy.py`
2. Add vocabulary loading and action decoding utilities
3. Support multi-task training with both action spaces
4. Add evaluation metrics specific to k_action predictions

---

## File Summary

| File | Changes |
|------|---------|
| `cfgs/dataset/waymo/base.yaml` | +4 lines (action_space, k_action_vocab_size, k_action_vocab_path) |
| `cfgs/model/base.yaml` | +3 lines (action_space, use_residual_mlp, residual_mlp_layers) |
| `utils/layers.py` | +46 lines (ResidualMLP class) |
| `modules/encoder.py` | +29/-9 lines (dual action dim, _build_mlp, k_actions selection) |
| `modules/decoder.py` | +50/-12 lines (dual heads, _reshape_actions, _build_mlp) |
| `models/ctrl_sim.py` | +18/-4 lines (action dim config, loss selection) |
| `datasets/rl_waymo/dataset.py` | +11/-2 lines (k_actions in select_relevant_agents) |
| `datasets/rl_waymo/dataset_ctrl_sim.py` | +54/-5 lines (k_actions loading/saving/filtering) |
| `policies/autoregressive_policy.py` | +4/-1 lines (k_action guard) |
