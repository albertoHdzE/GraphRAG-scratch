import math
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_latex_table(path: Path, latex: str) -> None:
    path.write_text(latex.strip() + "\n", encoding="utf-8")


def _fmt(x: float, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    return f"{float(x):.{nd}f}"


def build_tables(retrieval_df: pd.DataFrame, systems_df: pd.DataFrame, tables_dir: Path) -> None:
    _mkdir(tables_dir)

    r = retrieval_df.copy()
    retrieval_table = (
        "\\begin{tabular}{lrrrr}\n"
        "\\toprule\n"
        "Metric & Mean & Std & Min & Max \\\\\n"
        "\\midrule\n"
        f"Coverage & {_fmt(r['graph_coverage'].mean(), 3)} & {_fmt(r['graph_coverage'].std(), 3)} & {_fmt(r['graph_coverage'].min(), 3)} & {_fmt(r['graph_coverage'].max(), 3)} \\\\\n"
        f"Coherence & {_fmt(r['path_coherence'].mean(), 3)} & {_fmt(r['path_coherence'].std(), 3)} & {_fmt(r['path_coherence'].min(), 3)} & {_fmt(r['path_coherence'].max(), 3)} \\\\\n"
        f"Seed latency (ms) & {_fmt(r['seed_latency_ms'].mean(), 2)} & {_fmt(r['seed_latency_ms'].std(), 2)} & {_fmt(r['seed_latency_ms'].min(), 2)} & {_fmt(r['seed_latency_ms'].max(), 2)} \\\\\n"
        f"Traversal latency (ms) & {_fmt(r['graph_latency_ms'].mean(), 2)} & {_fmt(r['graph_latency_ms'].std(), 2)} & {_fmt(r['graph_latency_ms'].min(), 2)} & {_fmt(r['graph_latency_ms'].max(), 2)} \\\\\n"
        f"Expanded nodes & {_fmt(r['graph_nodes'].mean(), 1)} & {_fmt(r['graph_nodes'].std(), 1)} & {_fmt(r['graph_nodes'].min(), 1)} & {_fmt(r['graph_nodes'].max(), 1)} \\\\\n"
        f"Expanded edges & {_fmt(r['graph_edges'].mean(), 1)} & {_fmt(r['graph_edges'].std(), 1)} & {_fmt(r['graph_edges'].min(), 1)} & {_fmt(r['graph_edges'].max(), 1)} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    _write_latex_table(tables_dir / "metrics_summary.tex", retrieval_table)

    s = systems_df.copy()
    s["latency_ms"] = pd.to_numeric(s["latency_ms"], errors="coerce")
    s["info_density"] = pd.to_numeric(s["info_density"], errors="coerce")
    s["faithfulness"] = pd.to_numeric(s["faithfulness"], errors="coerce")
    s["efficiency"] = pd.to_numeric(s["efficiency"], errors="coerce")

    summary_rows = []
    for system in ["rag", "rau", "graphrag"]:
        block = s[s["system"] == system]
        summary_rows.append(
            {
                "system": system,
                "latency_ms_mean": float(block["latency_ms"].mean()),
                "latency_ms_std": float(block["latency_ms"].std()),
                "faithfulness_mean": float(block["faithfulness"].mean()),
                "faithfulness_std": float(block["faithfulness"].std()),
                "info_density_mean": float(block["info_density"].mean()),
                "info_density_std": float(block["info_density"].std()),
            }
        )
    summ = pd.DataFrame(summary_rows)

    systems_table = (
        "\\begin{tabular}{lrrrrrr}\n"
        "\\toprule\n"
        "System & Latency mean & Latency std & Faith mean & Faith std & Density mean & Density std \\\\\n"
        "\\midrule\n"
    )
    for _, row in summ.iterrows():
        systems_table += (
            f"{row['system']} & {_fmt(row['latency_ms_mean'], 1)} & {_fmt(row['latency_ms_std'], 1)} & "
            f"{_fmt(row['faithfulness_mean'], 2)} & {_fmt(row['faithfulness_std'], 2)} & "
            f"{_fmt(row['info_density_mean'], 3)} & {_fmt(row['info_density_std'], 3)} \\\\\n"
        )
    systems_table += "\\bottomrule\n\\end{tabular}\n"
    _write_latex_table(tables_dir / "systems_summary.tex", systems_table)


def build_plots(retrieval_df: pd.DataFrame, systems_df: pd.DataFrame, images_dir: Path) -> None:
    _mkdir(images_dir)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=retrieval_df, x="graph_coverage", color="#90caf9")
    plt.xlabel("graph_coverage")
    plt.tight_layout()
    plt.savefig(images_dir / "coverage_boxplot.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=retrieval_df, x="path_coherence", color="#90caf9")
    plt.xlabel("path_coherence")
    plt.tight_layout()
    plt.savefig(images_dir / "coherence_boxplot.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=retrieval_df, x="seed_latency_ms", color="#90caf9")
    plt.xlabel("seed_latency_ms")
    plt.tight_layout()
    plt.savefig(images_dir / "seed_latency_boxplot.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=retrieval_df, x="graph_latency_ms", color="#90caf9")
    plt.xlabel("graph_latency_ms")
    plt.tight_layout()
    plt.savefig(images_dir / "traversal_latency_boxplot.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6.5, 5))
    sns.scatterplot(data=retrieval_df, x="graph_nodes", y="graph_edges", color="#90caf9")
    plt.xlabel("graph_nodes")
    plt.ylabel("graph_edges")
    plt.tight_layout()
    plt.savefig(images_dir / "nodes_edges_scatter.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6.5, 5))
    sns.scatterplot(data=retrieval_df, x="graph_coverage", y="path_coherence", color="#90caf9")
    plt.xlabel("graph_coverage")
    plt.ylabel("path_coherence")
    plt.tight_layout()
    plt.savefig(images_dir / "coverage_vs_coherence.png", dpi=160)
    plt.close()

    s = systems_df.copy()
    s["latency_ms"] = pd.to_numeric(s["latency_ms"], errors="coerce")
    s["faithfulness"] = pd.to_numeric(s["faithfulness"], errors="coerce")
    s["info_density"] = pd.to_numeric(s["info_density"], errors="coerce")

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=s, x="system", y="latency_ms", color="#90caf9")
    plt.xlabel("system")
    plt.ylabel("latency_ms")
    plt.tight_layout()
    plt.savefig(images_dir / "latency_by_system_boxplot.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=s, x="system", y="faithfulness", color="#90caf9")
    plt.xlabel("system")
    plt.ylabel("faithfulness")
    plt.tight_layout()
    plt.savefig(images_dir / "faithfulness_by_system_boxplot.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=s, x="system", y="info_density", color="#90caf9")
    plt.xlabel("system")
    plt.ylabel("info_density")
    plt.tight_layout()
    plt.savefig(images_dir / "density_by_system_boxplot.png", dpi=160)
    plt.close()


def main() -> None:
    base = Path(__file__).resolve().parent
    retrieval_csv = base / "results_metrics.csv"
    systems_csv = base / "results_systems.csv"

    retrieval_df = pd.read_csv(retrieval_csv)
    systems_df = pd.read_csv(systems_csv)

    build_tables(retrieval_df, systems_df, base / "tables")
    build_plots(retrieval_df, systems_df, base / "images")
    print("Artifacts generated")


if __name__ == "__main__":
    main()

