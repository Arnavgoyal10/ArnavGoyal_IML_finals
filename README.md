# ArnavGoyal_IML_finals

## Repository Structure and Workflow

### Data Processing Pipeline
- `stage1.py` and `stage2.py`: These scripts handle the conversion of genomic data to proteomic sequences, creating our application dataset. The pipeline processes raw genomic data and transforms it into a format suitable for protein sequence analysis.

### Model Architecture
- `katransformer.py`: Implements the core transformer architecture with a rational activation function. This custom implementation is specifically designed for protein sequence analysis, incorporating domain-specific modifications to better capture biological patterns.

### Training and Optimization
- `train.py`: Contains the training loop and model training logic. This script handles the model training process, including data loading, training iterations, and model checkpointing.
- `hyperparameter_optimisation.py`: Implements Bayesian optimization for transformer parameters. This script systematically searches the hyperparameter space to find optimal model configurations, improving model performance through data-driven parameter selection.

