from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
from datetime import date
from pathlib import Path

import markdown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from pypdf import PdfReader
from weasyprint import CSS, HTML


MODEL_ORDER = [
    "microsoft/MiniLM-L12-H384-uncased",
    "google/electra-small-discriminator",
    "microsoft/deberta-v3-small",
    "Qwen/Qwen2.5-1.5B-Instruct",
]
SHORT_NAMES = {
    MODEL_ORDER[0]: "MiniLM-L12-H384",
    MODEL_ORDER[1]: "ELECTRA-small",
    MODEL_ORDER[2]: "DeBERTa-v3-small",
    MODEL_ORDER[3]: "Qwen2.5-1.5B",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the four-model GPU experiment report")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser


def _read_inputs(root: Path) -> tuple[pd.DataFrame, dict[str, dict], dict]:
    metrics_path = root / "artifacts/reports/run_metrics.csv"
    audit_path = root / "artifacts/data_audit.json"
    if not metrics_path.exists() or not audit_path.exists():
        raise FileNotFoundError("Run benchmark report and data audit before building this report")
    metrics = pd.read_csv(metrics_path)
    metrics = metrics[(metrics.scenario == "full") & metrics.model_id.isin(MODEL_ORDER)].copy()
    if len(metrics) != 4 or set(metrics.model_id) != set(MODEL_ORDER):
        raise ValueError("The detailed report requires exactly the four expected full-field runs")
    metadata: dict[str, dict] = {}
    for path in sorted((root / "artifacts").glob("**/metadata.json")):
        item = json.loads(path.read_text())
        if item.get("model_id") in MODEL_ORDER:
            metadata[item["model_id"]] = item
    if set(metadata) != set(MODEL_ORDER):
        raise ValueError("Missing metadata for one or more expected models")
    audit = json.loads(audit_path.read_text())
    return metrics.set_index("model_id").loc[MODEL_ORDER].reset_index(), metadata, audit


def _validate_predictions(root: Path, metrics: pd.DataFrame) -> None:
    for row in metrics.itertuples(index=False):
        path = root / row.prediction_file
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if len(frame) != 450 or set(frame.split) != {"test"}:
            raise ValueError(f"{row.model_id} must contain exactly 450 test predictions")
        valid = frame.valid_output.astype(str).str.lower().isin({"true", "1"})
        if int(valid.sum()) != int(row.n):
            raise ValueError(f"Valid-output count mismatch for {row.model_id}")
        tn = int(((frame.label == 0) & (frame.prediction == 0) & valid).sum())
        fp = int(((frame.label == 0) & (frame.prediction == 1) & valid).sum())
        fn = int(((frame.label == 1) & (frame.prediction == 0) & valid).sum())
        tp = int(((frame.label == 1) & (frame.prediction == 1) & valid).sum())
        if (tn, fp, fn, tp) != (row.tn, row.fp, row.fn, row.tp):
            raise ValueError(f"Confusion matrix mismatch for {row.model_id}")


def _style() -> None:
    font_path = subprocess.check_output(
        ["fc-match", "-f", "%{file}", "Noto Sans CJK SC"], text=True
    ).strip()
    font_manager.fontManager.addfont(font_path)
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({
        "font.family": font_name,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "#f8fafc",
        "axes.edgecolor": "#94a3b8",
        "grid.color": "#dbe4ee",
    })


def _generate_figures(root: Path, metrics: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    _style()
    colors = ["#2563eb", "#0f766e", "#7c3aed", "#ea580c"]
    labels = [SHORT_NAMES[item] for item in metrics.model_id]
    for metric, title in [
        ("accuracy", "Accuracy 对比"), ("precision", "Precision 对比"),
        ("recall", "Recall 对比"), ("f1", "F1 对比"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5.4))
        values = metrics[metric].to_numpy()
        bars = ax.bar(labels, values, color=colors, width=0.65)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric.title())
        ax.set_title(title, weight="bold")
        ax.grid(axis="y", alpha=.7)
        ax.bar_label(bars, fmt="%.3f", padding=3)
        fig.tight_layout()
        fig.savefig(output / f"{metric}.png", dpi=220)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
    for ax, row, color in zip(axes.flat, metrics.itertuples(index=False), colors):
        matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]])
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=metrics[["tn", "fp", "fn", "tp"]].to_numpy().max())
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center", fontsize=15, weight="bold",
                    color="white" if value > matrix.max() * .55 else "#172033")
        ax.set_title(SHORT_NAMES[row.model_id], color=color, weight="bold")
        ax.set_xticks([0, 1], ["预测 NO_MATCH", "预测 MATCH"])
        ax.set_yticks([0, 1], ["实际 NO_MATCH", "实际 MATCH"])
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.65, label="样本数")
    fig.suptitle("四模型混淆矩阵（锁定测试集 n=450）", fontsize=16, weight="bold")
    fig.savefig(output / "confusion_matrices.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    latency = axes[0].bar(labels, metrics.latency_p50_ms, color=colors)
    axes[0].bar_label(latency, fmt="%.1f ms", padding=3)
    axes[0].set_title("p50 单样本延迟")
    throughput = axes[1].bar(labels, metrics.throughput_per_second, color=colors)
    axes[1].bar_label(throughput, fmt="%.1f/s", padding=3)
    axes[1].set_title("吞吐量")
    for ax in axes:
        ax.grid(axis="y", alpha=.7)
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output / "efficiency.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for row, color in zip(metrics.itertuples(index=False), colors):
        ax.scatter(row.latency_p50_ms, row.f1, s=180, color=color, edgecolor="white", linewidth=1.5)
        ax.annotate(SHORT_NAMES[row.model_id], (row.latency_p50_ms, row.f1), xytext=(7, 5),
                    textcoords="offset points", fontsize=10)
    ax.set_xlabel("p50 延迟（ms，越低越好）")
    ax.set_ylabel("F1（越高越好）")
    ax.set_title("质量—延迟权衡", weight="bold")
    ax.grid(alpha=.7)
    fig.tight_layout()
    fig.savefig(output / "quality_latency_tradeoff.png", dpi=220)
    plt.close(fig)


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _result_table(metrics: pd.DataFrame) -> str:
    lines = [
        "| 模型 | Accuracy | Precision | Recall | F1 | F1 95% CI | Balanced Acc. | MCC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.sort_values("f1", ascending=False).itertuples(index=False):
        lines.append(
            f"| {SHORT_NAMES[row.model_id]} | {row.accuracy:.4f} | {row.precision:.4f} | "
            f"{row.recall:.4f} | **{row.f1:.4f}** | [{row.f1_ci95_low:.4f}, {row.f1_ci95_high:.4f}] | "
            f"{row.balanced_accuracy:.4f} | {row.mcc:.4f} |"
        )
    return "\n".join(lines)


def _confusion_table(metrics: pd.DataFrame) -> str:
    lines = ["| 模型 | TN | FP | FN | TP | 有效数 | 无效输出率 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in metrics.itertuples(index=False):
        lines.append(f"| {SHORT_NAMES[row.model_id]} | {row.tn} | {row.fp} | {row.fn} | {row.tp} | {row.n} | {_pct(row.invalid_output_rate)} |")
    return "\n".join(lines)


def _efficiency_table(metrics: pd.DataFrame, metadata: dict[str, dict]) -> str:
    lines = [
        "| 模型 | p50 (ms) | p95 (ms) | 吞吐量 (/s) | 平均输入 token | 平均输出 token | 峰值显存 (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        peak = metadata[row.model_id]["peak_gpu_memory_mb"]
        lines.append(f"| {SHORT_NAMES[row.model_id]} | {row.latency_p50_ms:.2f} | {row.latency_p95_ms:.2f} | {row.throughput_per_second:.2f} | {row.average_input_tokens:.2f} | {row.average_output_tokens:.2f} | {peak:.2f} |")
    return "\n".join(lines)


def _model_table(metadata: dict[str, dict]) -> str:
    lines = [
        "| 模型 | 赛道/制度 | 最佳 LR | 训练精度 | 模型 revision |",
        "|---|---|---:|---|---|",
    ]
    for model_id in MODEL_ORDER:
        item = metadata[model_id]
        is_prompt = model_id.startswith("Qwen/")
        lr = "—" if is_prompt else f"{item['best_learning_rate']:.0e}"
        precision = "BF16 推理" if is_prompt else (item.get("training_precision") or "bf16").upper()
        regime = "提示推理 / zero-shot" if is_prompt else "监督分类 / fine-tuned"
        lines.append(f"| {SHORT_NAMES[model_id]} | {regime} | {lr} | {precision} | `{item['model_revision'][:12]}` |")
    return "\n".join(lines)


def _report(metrics: pd.DataFrame, metadata: dict[str, dict], audit: dict) -> str:
    best = metrics.sort_values("f1", ascending=False).iloc[0]
    qwen = metrics[metrics.model_id == MODEL_ORDER[3]].iloc[0]
    electra = metrics[metrics.model_id == MODEL_ORDER[1]].iloc[0]
    deberta = metrics[metrics.model_id == MODEL_ORDER[2]].iloc[0]
    package = metadata[MODEL_ORDER[0]]["packages"]
    generated = date.today().isoformat()
    return f"""# 四模型 GPU 地点匹配实验详细报告

> 报告类型：中文技术评审报告  
> 生成日期：{generated}  
> 硬件：{metadata[MODEL_ORDER[0]]['gpu']}  
> 数据来源：四份逐样本预测、统一指标汇总和模型运行 metadata

[TOC]

## 摘要

本实验面向 Overture 风格地点记录的二分类匹配任务：判断两条地点记录是否指向同一现实实体。为修复原 Notebook 中字段删除无效、FP/FN 记账对调及随机切分可能造成实体泄漏等可信度问题，本次将流程重构为配置驱动 CLI，并在统一、锁定的数据切分上完成三个 Encoder 微调实验和一个小型生成模型 zero-shot 实验。

共审计 {audit['raw_rows']} 条原始样本，去除 {audit['duplicate_pair_rows_removed']} 条重复后保留 {audit['clean_rows']} 条；按实体连通分量划分为训练 {audit['splits']['train']['rows']}、验证 {audit['splits']['validation']['rows']}、测试 {audit['splits']['test']['rows']} 条。四个模型均在同一 450 条测试集上评估。主指标 F1 最优模型为 **{SHORT_NAMES[best.model_id]}（{best.f1:.4f}，95% CI [{best.f1_ci95_low:.4f}, {best.f1_ci95_high:.4f}]）**。

核心工程结论是：监督 Encoder 在本任务上优于直接 zero-shot 生成推理；MiniLM 同时取得最高 Accuracy、F1 和 MCC，并具有最低 p50 延迟。Qwen 达到 100% Recall，但以 {int(qwen.fp)} 个假阳性为代价，不能仅凭 F1 将其视为稳健的实体合并器。

## 1. 本次改动

### 1.1 从 Notebook 到可复现实验框架

- 将数据审计、切分、Encoder 训练、提示推理、评估与报告整理为统一 `benchmark` CLI。
- 所有实验由配置、模型 ID、场景、随机种子和 regime 明确标识；逐样本保存预测、原始输出、token 与延迟。
- 保存 Python、PyTorch、Transformers、CUDA/GPU、Git revision 与 Hugging Face 模型 revision。
- 可视化只从预测和汇总 CSV 生成，不在 Notebook 或报告中维护独立指标数组。

### 1.2 影响结论可信度的问题修复

1. **指标方向修复**：删除原先手工且对调 FP/FN 的逻辑，统一由 `sklearn` 计算混淆矩阵、Precision、Recall、F1、Balanced Accuracy 与 MCC。
2. **输入泄漏修复**：通过序列化器显式选择名称、类别、网站、社交账号、邮箱、电话、品牌和地址；`sources`、`confidence`、`id`、`base_id` 不进入模型文本。
3. **实体泄漏修复**：用 `id`/`base_id` 建立关系图，按连通分量分组切分，确保同一实体 ID 不跨训练、验证和测试集合。
4. **测试集锁定**：提示选择和学习率选择仅使用训练/验证数据；测试集只在最佳 checkpoint 确定后评估。
5. **环境兼容修复**：移除绝对 Colab 路径、硬编码保存位置、无条件 BF16 和模型专属 tokenizer 假设。
6. **历史结果隔离**：原 Notebook 结果保留用于溯源，但标为历史且不可验证，不进入新排行榜。

### 1.3 DeBERTa 数值稳定性修复

在当前 Transformers {package['transformers']} 环境中，DeBERTa checkpoint 的参数实际被物化为 FP16，即使训练器混合精度已关闭，AdamW 第一步仍会产生非有限参数。短步诊断确认显式 `dtype=torch.float32` 后连续 20 次更新均保持有限，因此正式 DeBERTa 训练固定使用 FP32。此修复属于运行兼容性措施，不改变数据或标签。

## 2. 实验整体框架

实验数据流如下：

```text
只读 Parquet → 数据审计/去重 → 实体关系图 → 70/15/15 分组切分
                                      ↓
统一字段序列化 → Encoder 微调（LR 选择） ─┬→ 锁定测试集预测
                 Qwen zero-shot 提示 ────┘
                                      ↓
sklearn 指标 → 1000 次分层 bootstrap → CSV/图表/本报告
```

### 2.1 数据与防泄漏

| 集合 | 样本数 | 正例数 | 正例比例 |
|---|---:|---:|---:|
| 训练 | {audit['splits']['train']['rows']} | {audit['splits']['train']['positives']} | {_pct(audit['splits']['train']['positives']/audit['splits']['train']['rows'])} |
| 验证 | {audit['splits']['validation']['rows']} | {audit['splits']['validation']['positives']} | {_pct(audit['splits']['validation']['positives']/audit['splits']['validation']['rows'])} |
| 测试 | {audit['splits']['test']['rows']} | {audit['splits']['test']['positives']} | {_pct(audit['splits']['test']['positives']/audit['splits']['test']['rows'])} |

测试集正例占比为 {_pct(audit['splits']['test']['positives']/audit['splits']['test']['rows'])}，因此仅看 Accuracy 容易掩盖类别倾向。本报告同时使用 F1、Balanced Accuracy、MCC 和完整混淆矩阵。

### 2.2 两条评测赛道

- **监督分类赛道**：ELECTRA、MiniLM、DeBERTa 使用 `AutoModelForSequenceClassification(num_labels=2)` 全参数微调；最大长度 256、batch size 32、weight decay 0.01、最多 10 epochs、验证 F1 早停 patience 2。每个模型比较学习率 `2e-5` 与 `5e-5`，正式运行 seed 42。
- **提示推理赛道**：Qwen2.5-1.5B-Instruct 使用官方 chat template、zero-shot、确定性解码、最多 4 个新 token；只接受 `MATCH` 或 `NO_MATCH`，额外文本记为 invalid。

{_model_table(metadata)}

## 3. 实验结果

### 3.1 综合质量指标

{_result_table(metrics)}

![四模型 F1 对比](assets/f1.png)

从主指标看，MiniLM 以 {best.f1:.4f} 排名第一，比 ELECTRA 高 {best.f1-electra.f1:.4f}，比 DeBERTa 高 {best.f1-deberta.f1:.4f}。三者 95% bootstrap 区间存在重叠，因此本轮单 seed 结果支持“MiniLM 是当前最佳候选”，但不足以断言模型架构间存在稳定的统计显著差异。

![Accuracy 对比](assets/accuracy.png)

![Precision 对比](assets/precision.png)

![Recall 对比](assets/recall.png)

### 3.2 混淆矩阵与错误倾向

{_confusion_table(metrics)}

![四模型混淆矩阵](assets/confusion_matrices.png)

- **MiniLM**：FP 与 FN 分别为 {int(best.fp)} 和 {int(best.fn)}，在正负类之间最均衡，MCC {best.mcc:.4f} 为四模型最高。
- **ELECTRA**：Recall {electra.recall:.4f}，仅漏掉 {int(electra.fn)} 个正例，但产生 {int(electra.fp)} 个假阳性，适合“宁可多召回、后续再复核”的流程。
- **DeBERTa**：Precision {deberta.precision:.4f} 为四模型最高，FP 仅 {int(deberta.fp)}；代价是 FN 增至 {int(deberta.fn)}，适合误合并成本高的场景。
- **Qwen**：把全部 {audit['splits']['test']['positives']} 个正例识别为 MATCH，却将 {int(qwen.fp)} 个负例误判为 MATCH；Balanced Accuracy 仅 {qwen.balanced_accuracy:.4f}。它表现出强烈正类偏置，而不是均衡的实体消歧能力。

### 3.3 延迟、吞吐量与资源

{_efficiency_table(metrics, metadata)}

![效率对比](assets/efficiency.png)

Encoder 的 p50 延迟为 {metrics.latency_p50_ms.min():.2f}–{metrics[metrics.track=='encoder'].latency_p50_ms.max():.2f} ms，Qwen 为 {qwen.latency_p50_ms:.2f} ms。MiniLM 同时取得最高 F1 和最低延迟，位于本轮质量—效率的优势区域。Qwen 的生成式解码使延迟约为 MiniLM 的 {qwen.latency_p50_ms/best.latency_p50_ms:.1f} 倍，峰值显存约 {metadata[MODEL_ORDER[3]]['peak_gpu_memory_mb']/1024:.2f} GiB。

![质量与延迟权衡](assets/quality_latency_tradeoff.png)

## 4. 图文结论与模型选择建议

1. **默认部署候选：MiniLM**。其 F1、Accuracy、MCC 均最高，延迟最低，适合作为当前地点匹配主模型。
2. **召回优先候选：ELECTRA**。若漏掉真实匹配的代价更高，可利用其较高 Recall，并在后处理阶段控制 FP。
3. **精度优先候选：DeBERTa**。当错误合并会污染主数据时，其最高 Precision 和最低 FP 更有吸引力；但需接受 FP32 训练和更高漏检。
4. **Qwen 不宜直接做自动合并器**。zero-shot 可以作为高召回候选生成器或规则/Encoder 之后的辅助信号，但本轮正类偏置过强，不应仅按 0.7959 的 F1 作正面判断。

## 5. 可信度、限制与解释边界

### 5.1 已完成的可信度保障

- 同一份 split、字段顺序、最大长度、选择指标和计时方式用于可比较实验。
- 每个测试结论均可追溯至 450 条逐样本预测；混淆矩阵总数与有效预测数一致。
- 对测试预测执行 1000 次分层 bootstrap，报告 95% 区间。
- Qwen 严格解析输出，本轮 invalid rate 为 {_pct(qwen.invalid_output_rate)}，无人工修正。
- 本地资源消耗与托管 API 价格分开；本报告不输出不可核验的美元成本。

### 5.2 本轮限制

- Encoder 仅运行 seed 42，尚未获得三随机种子的均值与标准差；模型排序可能受初始化波动影响。
- 未执行字段消融，不能从本轮结果推断 email、website、address 等字段的独立贡献。
- Qwen 仅为 zero-shot；未测试固定 3-shot，也未进行阈值校准或微调。
- Llama 3.2 与 Gemma 2 因 Hugging Face gated access 被主动跳过。
- 测试集仅 450 条且来自单一数据来源，跨地区、语言和类别的外部有效性仍需额外数据验证。
- 延迟来自单张 RTX A6000、batch size 1，不代表其他 GPU、CPU 或批处理部署的性能。

## 6. 复现方式

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,report]'

benchmark validate-data
benchmark train-encoder --model google/electra-small-discriminator --scenario full --seed 42
benchmark train-encoder --model microsoft/MiniLM-L12-H384-uncased --scenario full --seed 42
benchmark train-encoder --model microsoft/deberta-v3-small --scenario full --seed 42
benchmark run-prompt --model Qwen/Qwen2.5-1.5B-Instruct --regime zero --scenario full
benchmark report
python scripts/build_experiment_report.py
```

模型缓存、checkpoint 和虚拟环境不进入版本库；发布的预测、metadata、split manifest 与汇总表足以追溯本报告中的所有数字。

## 7. 结论

本轮工作不仅补充了四模型 GPU 实验，也修复了会系统性影响旧结论的评估问题。基于锁定测试集，MiniLM 是质量与效率最均衡的默认候选；ELECTRA 与 DeBERTa 分别提供召回优先和精度优先的替代方案；Qwen zero-shot 暴露出明显的 MATCH 偏置。下一阶段最有价值的工作是完成 Encoder 三 seed 重复实验、字段消融以及 Qwen 固定 3-shot，对当前排序的稳定性和字段依赖做进一步验证。

---

数据来源：`artifacts/reports/run_metrics.csv`、四份 `predictions.csv`、四份 `metadata.json` 与 `artifacts/data_audit.json`。报告和图表由 `scripts/build_experiment_report.py` 自动生成。
"""


def _build_pdf(root: Path, markdown_path: Path, css_path: Path, pdf_path: Path) -> None:
    source = markdown_path.read_text()
    body = markdown.markdown(source, extensions=["tables", "fenced_code", "toc"], output_format="html5")
    title = "四模型 GPU 地点匹配实验详细报告"
    cover = (
        '<section class="cover"><h1>' + html.escape(title) + '</h1>'
        '<div class="subtitle">可复现的小模型地点匹配技术评审</div>'
        '<div class="meta">RTX A6000 · 三个监督 Encoder · Qwen zero-shot<br>'
        + date.today().isoformat() + '</div></section>'
    )
    document = f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>{title}</title></head><body>{cover}{body}</body></html>"
    HTML(string=document, base_url=str(root / "docs")).write_pdf(pdf_path, stylesheets=[CSS(filename=css_path)])


def _verify_pdf(pdf_path: Path) -> None:
    reader = PdfReader(pdf_path)
    if len(reader.pages) < 8 or pdf_path.stat().st_size < 200_000:
        raise ValueError("Generated PDF is unexpectedly small or short")
    text = "".join((page.extract_text() or "") for page in reader.pages)
    for phrase in ("四模型 GPU 地点匹配实验详细报告", "MiniLM", "可信度", "结论"):
        if phrase not in text:
            raise ValueError(f"PDF text verification failed: {phrase}")
    image_count = sum(len(page.images) for page in reader.pages)
    if image_count < 7:
        raise ValueError(f"Expected at least 7 embedded report figures, found {image_count}")


def main() -> None:
    root = _parser().parse_args().root.resolve()
    docs = root / "docs"
    assets = docs / "assets"
    markdown_path = docs / "four-model-gpu-experiment-report.md"
    pdf_path = docs / "four-model-gpu-experiment-report.pdf"
    css_path = docs / "report.css"
    metrics, metadata, audit = _read_inputs(root)
    _validate_predictions(root, metrics)
    if not shutil.which("fc-match") or "Noto" not in subprocess.check_output(
        ["fc-match", "Noto Sans CJK SC", "family"], text=True
    ):
        raise RuntimeError("Noto CJK font is required; install fonts-noto-cjk")
    _generate_figures(root, metrics, assets)
    markdown_path.write_text(_report(metrics, metadata, audit))
    _build_pdf(root, markdown_path, css_path, pdf_path)
    _verify_pdf(pdf_path)
    print(markdown_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
