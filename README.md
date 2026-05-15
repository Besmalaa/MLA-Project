# Topic 5 – Continual Learning against Catastrophic Forgetting in Dermatology

## Course: Advanced Machine Learning 2025/2026

### Objective
Learn to classify new skin diseases **sequentially** without forgetting previously learned ones (class-incremental learning) using the **DermaMNIST** dataset from MedMNIST.

---

## Project Structure

```
continual_learning_dermatology/
├── README.md
├── requirements.txt
├── data/                        # Downloaded automatically by medmnist
├── notebooks/
│   ├── W1_project_scope.ipynb
│   ├── W2_data_exploration.ipynb
│   └── W3_experiments_summary.ipynb
├── src/
│   ├── models/cnn.py            # CNN backbone
│   ├── methods/                 # naive, ewc, lwf, replay, joint
│   └── utils/                  # data, metrics, trainer
├── scripts/
│   └── run_all_experiments.py
└── results/
```

## Setup
```bash
pip install -r requirements.txt
```

## Reproduce All Results
```bash
python scripts/run_all_experiments.py
```

## Methods Compared
| Method            | Type            | Role       |
|-------------------|-----------------|------------|
| Naive Fine-tuning | Baseline        | Lower bound (forgetting) |
| EWC               | Regularization  | Main method |
| LwF               | Regularization  | Main method |
| Experience Replay | Memory-based    | Main method |
| Joint Training    | Oracle          | Upper bound |

## Metrics
- **FAA**: Final Average Accuracy across all tasks
- **BWT**: Backward Transfer (negative = forgetting)
