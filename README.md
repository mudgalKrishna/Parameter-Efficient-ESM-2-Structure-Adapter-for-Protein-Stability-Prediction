# Protein Stability Prediction with ESM-2 and a Structure Adapter

This project investigates whether a machine-learning model can estimate how a mutation (a change in one amino acid) affects the stability of a protein. It combines a protein language model, ESM-2, with information from protein structure and evaluates the predictions on the Tsuboyama protein stability benchmark.

The goal is to make useful predictions from existing experimental data—not to claim that a computer can replace laboratory testing.

## 1. Why predict protein stability?

Proteins are long chains of amino acids that fold into particular three-dimensional shapes. Their shape and stability affect how they work. A small change in the amino-acid sequence can make a protein more stable, less stable, or have little effect.

Researchers can measure these effects experimentally, but testing many possible mutations can take time and resources. A prediction model can help estimate which mutations may be worth investigating further.

## 2. What does this project predict?

The project predicts **ΔΔG (delta-delta G)**, a value used to describe the change in protein stability caused by a mutation. In this benchmark, the convention is:

`ΔΔG = ΔG_mutant − ΔG_wild-type`

Under this convention, a negative value indicates a destabilizing mutation. The model learns from examples with experimental measurements and predicts a value for each mutation.

## 3. How the model works

The best-performing configuration in our experiments combines ESM-2 650M, LoRA, ProteinMPNN, and a structure adapter. Each part has a different role:

- **ESM-2:** A protein language model. It has learned patterns from large collections of protein sequences and turns a sequence into numerical representations that a prediction model can use.
- **LoRA:** A parameter-efficient way to adapt a large model. Instead of updating all of ESM-2, LoRA adds a smaller set of trainable parameters. This can reduce the number of parameters that need to be updated during training.
- **ProteinMPNN:** A pretrained model designed around protein structure. In this project, its encoder provides additional information from the protein's backbone structure.
- **Structure adapter:** A small trainable component that combines the sequence information from ESM-2 with the structural information from ProteinMPNN.
- **Prediction head:** The final regression layer that turns the learned representation into a numerical ΔΔG prediction.

ESM-2 provides sequence information, ProteinMPNN provides structure information, and the prediction layers learn how these relate to measured changes in stability.

We also tested sequence-only models, models with different training strategies, simpler feature-based baselines, and homology-filtered experiments to understand which parts of the approach were useful.

## 4. Dataset and evaluation setup

We used the **Tsuboyama ΔΔG benchmark**, which contains experimental measurements for protein variants.

| Split | Proteins | Variants | Purpose |
| --- | ---: | ---: | --- |
| Training | 239 | 215,731 | Used to train model parameters |
| Validation | 31 | 27,328 | Used to compare configurations and choose checkpoints |
| Test | 28 | 28,172 | Held back for final evaluation |

The split is made at the **protein level**: a protein does not appear in more than one of the three splits. This is important because variants from the same protein are related, and mixing one protein across splits could make evaluation look more optimistic than it should.

Model selection was based on the validation set. The final test set was kept separate from training and model selection, then used for the locked test evaluation.

## 5. How to understand the main metric

The main metric is **median per-protein Spearman correlation (ρ)**.

Spearman correlation measures whether the model ranks mutations in a similar order to the experimental measurements. For each test protein, we compare the predicted ranking with the measured ranking, then take the median score across the 28 test proteins.

- A score closer to **1** means the model's rankings generally agree better with the experimental rankings.
- A score around **0** means little rank agreement.
- A negative score means the rankings tend to go in opposite directions.

We use the median per-protein score as the headline result because variants from the same protein are not independent observations. The report also includes pooled metrics and error measures for completeness, but these are not the primary basis for comparing models.

## 6. Main test results

The table below shows the key results on the held-out test set. Higher median per-protein Spearman correlation is better.

| Model / configuration | Test median Spearman ρ | 95% confidence interval |
| --- | ---: | ---: |
| **ESM-2 650M + LoRA + ProteinMPNN + structure adapter** | **0.7547** | **0.7118–0.8137** |
| ESM-2 650M, full fine-tuning, sequence only | 0.7503 | 0.6900–0.8071 |
| ESM-2 150M + LoRA + ProteinMPNN + structure adapter | 0.7271 | 0.6610–0.7899 |
| ESM-2 150M, full fine-tuning, sequence only | 0.7231 | 0.6585–0.7921 |
| Ridge regression using 29 hand-crafted structure and sequence features | 0.6289 | 0.5860–0.6569 |
| ThermoMPNN released predictions (reference baseline) | 0.7720 | 0.7462–0.7993 |

**Best result from our experiments:** the ESM-2 650M model with LoRA and the structure branch achieved a test median Spearman correlation of **0.7547**.

### How does it compare with ThermoMPNN?

ThermoMPNN achieved **0.7720** on the same 28 test proteins. The paired median difference between our best model and ThermoMPNN was **−0.0173**, with a 95% confidence interval from **−0.0683 to +0.0383** and a Wilcoxon p-value of **0.465**.

This comparison does **not** show a statistically clear difference between the two models on this test set. We therefore describe our result as competitive with ThermoMPNN, not as an improvement over it.

## 7. What did the experiments teach us?

### The larger ESM-2 model performed better in this setup

For the structure-adapter configuration, the 650M model scored **0.7547**, compared with **0.7271** for the 150M model. The paired median difference was **+0.0276**, with a 95% confidence interval of **+0.0028 to +0.0736**. This supports a modest improvement from using the larger model on this benchmark.

### Fine-tuning was important

The trained ESM-2 models performed much better than the zero-shot ESM-2 scores and the simple Ridge baseline. This suggests that adapting the model to experimental stability data is important for this task.

### The structure branch added only a small extra gain

For ESM-2 650M, the structure-adapter model scored **0.7547**, compared with **0.7503** for the sequence-only full fine-tuning model. The median difference was **+0.0044**, and its confidence interval included zero. The observed gain is small, and this experiment does not establish a clear additional benefit from the structure branch over the sequence-only model.

### Models can make different predictions on individual proteins

Although the overall results of our best model and ThermoMPNN are close, they do not behave identically on every protein. Our model had a higher per-protein Spearman score on 12 of the 28 test proteins, while ThermoMPNN scored higher on 16. This suggests that future work could study whether complementary predictions from the two approaches can be combined. Such a combination is a possible next experiment, not something established by the current results.

## 8. Training details

The main experiments compared ESM-2 150M and ESM-2 650M. The final model predictions were formed by averaging predictions from eight independently seeded models.

| Setting | Value |
| --- | --- |
| ESM-2 backbones | `facebook/esm2_t30_150M_UR50D`, `facebook/esm2_t33_650M_UR50D` |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Loss function | Huber loss, β = 1.0 |
| Batch size | 16 |
| Maximum epochs | 100 |
| Early-stopping patience | 20 epochs |
| Optimizer weight decay | 0.01 |
| Random seeds | 42, 101, 202, 303, 404, 505, 606, 707 |
| Final prediction | Mean of eight seed-model predictions |
| Checkpoint selection metric | Validation median per-protein Spearman ρ |

The 650M structure-adapter model used a backbone learning rate of `1e-5` and a prediction-head learning rate of `1e-3`. The 150M model used a backbone learning rate of `2e-5` and a head learning rate of `1e-3`.

## 9. Baselines and additional experiments

To put the results in context, the experiments included:

- **Zero-shot ESM-2 scores:** predictions derived from the pretrained model without task-specific training.
- **Ridge regression:** a simpler baseline using 29 hand-crafted features related to the mutation and protein structure.
- **Sequence-only ESM-2 models:** models trained without the ProteinMPNN structure branch.
- **Structure-adapter models:** models that combine ESM-2 sequence representations and ProteinMPNN structure information.
- **Homology-filtered experiments:** additional 150M experiments that filtered training examples based on sequence identity to test proteins.
- **ThermoMPNN:** a reference baseline evaluated using its released predictions; it was not retrained as part of this project.

These comparisons help separate the benefit of training ESM-2 from the extra contribution of structural information and model size.

## 10. Repository files

The repository package contains this README and `updated_pipeline_bundle.zip`. The bundle includes the main scripts and experiment records, including:

- `train.py` and `train_esm2_spurs_fixed.py` — training entry points.
- `pipeline.py` — supporting pipeline code.
- `hill_climb_validation_log.csv` — validation sweep results.
- `test_once_final_report.csv` — final test metrics.
- `paired_bootstrap_tests_28_proteins.csv` — paired statistical comparisons across test proteins.
- `selected_model_before_test.json` — record of the selected model configuration before final test evaluation.
- `experiment_trace_manifest.json` and `esm2_finetune_audit_summary.json` — experiment and data-audit records.
- `separate_25_30_identity_experiment.csv` and `proteins_with_seq_identity_clusters.csv` — homology-related experiment records.

The ZIP bundle does not include the Word project report. The code and experiment records needed to inspect the reported results are listed above.

## 11. Reproducibility and data checks

Before training, the pipeline checked that the train, validation, and test proteins were separated; mutation labels matched the dataset records; and the structure inputs could be parsed. The recorded audit found no duplicate variants within the individual splits and no protein overlap between splits. The pretrained ESM-2 and ProteinMPNN weights were checked as part of the run setup.

The training and evaluation records include validation configurations, selected checkpoints, test metrics, and paired comparisons. These records are included to make the reported results easier to inspect and reproduce.

## 12. Limitations

- The final test set contains **28 proteins**, so performance on other protein families may differ.
- A high rank correlation does not mean every predicted ΔΔG value is numerically exact.
- The best model did not clearly outperform ThermoMPNN in the paired comparison.
- The extra gain from adding the structure branch over the sequence-only model was small and not clearly resolved by the reported confidence interval.
- Sequence homology, protein type, experimental conditions, and dataset coverage can affect how well the model generalizes.
- The model provides estimates for research use; predictions should not be treated as experimental confirmation.

## 13. Possible future direction

One possible next step is to test whether combining predictions from ESM-2 and ThermoMPNN helps on proteins where their individual predictions differ. This would need a separate, carefully designed evaluation to determine whether combining them actually improves results on unseen proteins.

## Summary

This project trained and compared ESM-2-based models for predicting mutation-related protein stability changes. The best tested configuration—**ESM-2 650M with LoRA, ProteinMPNN, and a structure adapter**—achieved a median per-protein Spearman correlation of **0.7547** on the held-out test set. The experiments show the value of task-specific training and a modest advantage for the larger ESM-2 model, while the additional benefit of the structure branch was small. ThermoMPNN scored **0.7720** on the same test proteins, so the current results support a competitive comparison rather than a claim of superiority.
