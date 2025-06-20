## Overriding config parameters

Default parameters are used for various aspects of the benchmark, e.g., the batch sizes, learning rates. These default parameters can be overriden using the following syntaxes for both CLI and API uses.

```bash
thunder benchmark keep bach knn --task.pre_comp_emb_batch_size 123 \
                                --task.k_vals "[1, 2, 3]"
```

```python
import thunder
thunder.benchmark('hiboub',
                  'bach',
                  'knn',
                  **{'task.pre_comp_emb_batch_size': 123, 'task.k_vals': [1, 2, 3]})
```

## Overridable parameters
Here is a non exhaustive list of the parameters that you may want to override per task, as well as the type and a small description.

### Frozen linear probing
| Name | Type | Description |
|------|---|---|
| adaptation.batch_size | int | Batch size used for training. |
| adaptation.num_workers | int | Number of workers for the data loader. |
| adaptation.lr | list[int] | List of learning rates used for the grid search. |
| adaptation.weight_decay | list[int] | List of weight decays used for the grid search. |
| adaptation.epochs | int | Number of training epochs. |


### LoRA linear probing
| Name | Type | Description |
|------|---|---|
| adaptation.lora_rank | int | Rank for the LoRA adapter. |
| adaptation.lora_alpha | int | Alpha parameter for LoRA. |
| adaptation.batch_size | int | Batch size used for training. |
| adaptation.num_workers | int | Number of workers for the data loader. |
| adaptation.lr | list[int] | List of learning rates used for the grid search. |
| adaptation.weight_decay | list[int] | List of weight decays used for the grid search. |
| adaptation.epochs | int | Number of training epochs. |

### Adversarial attack
| Name | Type | Description |
|------|---|---|
| task.pre_comp_emb_batch_size | int | Batch size for precomputing the embeddings. |
| task.attack_batch_size | int | Batch size for the attacks. |
| task.nb_attack_images | int | Number of images to use. |
| task.attack.eps | float | Radius of the norm ball. |
| task.attack.alpha | float | Step size per PGD iteration. |
| task.attack.n_steps | int | Number of PGD iterations. |

### Alignment scoring
| Name | Type | Description |
|------|---|---|
| task.pre_comp_emb_batch_size | int | Batch size for precomputing the embeddings. |

### Image retrieval
| Name | Type | Description |
|------|---|---|
| task.pre_comp_emb_batch_size | int | Batch size for precomputing the embeddings. |
| task.k_vals | list[int] | Values of k to use. |

### K-nn
| Name | Type | Description |
|------|---|---|
| task.pre_comp_emb_batch_size | int | Batch size for precomputing the embeddings. |
| task.k_vals | list[int] | Values of k to use. |

### Precomputing embeddings
| Name | Type | Description |
|------|---|---|
| task.pre_comp_emb_batch_size | int | Batch size for precomputing the embeddings. |

### Simple shot
| Name | Type | Description |
|------|---|---|
| task.pre_comp_emb_batch_size | int | Batch size for precomputing the embeddings. |

### Transformation invariance
| Name | Type | Description |
|------|---|---|
| task.transformation_invariance_batch_size | int | Batch size for the transformations. |
| task.nb_images | int | Number of images to use. |