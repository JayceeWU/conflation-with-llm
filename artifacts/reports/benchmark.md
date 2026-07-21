# Reproducible Small-Model Benchmark

Generated only from prediction artifacts.

## Full-field runs

| model_id                           | track   | regime     |   seed |   accuracy |   precision |   recall |     f1 |   f1_ci95_low |   f1_ci95_high |   balanced_accuracy |    mcc |   invalid_output_rate |   latency_p50_ms |
|:-----------------------------------|:--------|:-----------|-------:|-----------:|------------:|---------:|-------:|--------------:|---------------:|--------------------:|-------:|----------------------:|-----------------:|
| microsoft/MiniLM-L12-H384-uncased  | encoder | fine-tuned |     42 |     0.8400 |      0.8516 |   0.8893 | 0.8700 |        0.8429 |         0.8948 |              0.8273 | 0.6632 |                0.0000 |          16.0742 |
| google/electra-small-discriminator | encoder | fine-tuned |     42 |     0.8111 |      0.7962 |   0.9225 | 0.8547 |        0.8322 |         0.8793 |              0.7825 | 0.6021 |                0.0000 |          20.6654 |
| microsoft/deberta-v3-small         | encoder | fine-tuned |     42 |     0.8089 |      0.8599 |   0.8155 | 0.8371 |        0.8039 |         0.8672 |              0.8072 | 0.6076 |                0.0000 |          17.4643 |
| Qwen/Qwen2.5-1.5B-Instruct         | prompt  | zero       |     42 |     0.6911 |      0.6610 |   1.0000 | 0.7959 |        0.7844 |         0.8102 |              0.6117 | 0.3843 |                0.0000 |         108.4206 |

## Traceability

Machine-readable source: `run_metrics.csv`; aggregate: `summary.csv`.
