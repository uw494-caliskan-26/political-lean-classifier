# political-lean-classifier

## Quick Run Commands

```powershell
# 1) Install dependencies
pip install -r requirements.txt

# 2) Fine-tune model (writes best model + logs)
python ./scripts/finetune.py

# 3) Run single-text prediction
python ./scripts/predict.py --text "Tax cuts for corporations will result in increased economic activity."

# 4) Plot fine-tuning metrics (separate graphs)
python ./scripts/plot_finetune_metrics.py

# 5) Plot label and prediction distributions
python ./scripts/plot_label_prediction_distribution.py
```

## Training logs

Run fine-tuning:

```powershell
python ./scripts/finetune.py
```

Generated logging artifacts:

- Unified metrics CSV (all Trainer log/eval payloads + key scalar columns): `./results/logs/training_metrics.csv`

## Distribution analysis

Plot fine-tuning metrics from CSV:

```powershell
python ./scripts/plot_finetune_metrics.py
```

Optional args:

```powershell
python ./scripts/plot_finetune_metrics.py --metrics-path ./results/logs/training_metrics.csv --output-dir ./results/metrics_plots --dpi 200
```

Outputs:

- `./results/metrics_plots/training_loss.png`
- `./results/metrics_plots/validation_loss.png`
- `./results/metrics_plots/learning_rate.png`
- `./results/metrics_plots/gradient_norm.png`

## Label vs prediction distribution

Run label and prediction distribution analysis:

```powershell
python ./scripts/plot_label_prediction_distribution.py --data-path ./data/clean_data.parquet --model-dir ./results/best_bias_model --output-dir ./results/distribution_plots
```

Outputs:

- `./results/distribution_plots/label_distribution.png`
- `./results/distribution_plots/prediction_distribution.png`
- `./results/distribution_plots/label_prediction_overlay.png`
- `./results/distribution_plots/distribution_summary.csv`

