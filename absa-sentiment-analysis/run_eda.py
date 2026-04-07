"""
ABSA EDA Script — Run this on YOUR system.

Usage (from the project root, i.e. the folder containing data/):
    python run_eda.py

Results are saved to:
    results/figures/   → PNG charts
    results/metrics/   → CSV summary files
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent          # folder where this script lives
DATA_DIR  = ROOT / "data" / "raw"
FIG_DIR   = ROOT / "results" / "figures"
MET_DIR   = ROOT / "results" / "metrics"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MET_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
def load_split(domain: str) -> pd.DataFrame:
    train = pd.read_csv(DATA_DIR / f"{domain}_train.csv")
    test  = pd.read_csv(DATA_DIR / f"{domain}_test.csv")
    train["split"] = "train"
    test["split"]  = "test"
    df = pd.concat([train, test], ignore_index=True)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # normalise column names to expected names
    df = df.rename(columns={"sentence": "sentence",
                             "aspect_term": "aspect_term",
                             "polarity": "polarity"})
    df["domain"] = domain
    return df

print("Loading data …")
laptops     = load_split("laptops")
restaurants = load_split("restaurants")
df = pd.concat([laptops, restaurants], ignore_index=True)
print(f"  Total samples: {len(df):,}  |  columns: {list(df.columns)}")

# ── Helper ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")
PALETTE = {"positive": "#4CAF50", "negative": "#F44336", "neutral": "#2196F3",
           "conflict": "#FF9800"}


def save(fig: plt.Figure, name: str):
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Dataset size summary
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1/7] Dataset size summary")
size_df = (df.groupby(["domain", "split"])
             .size()
             .reset_index(name="samples"))
print(size_df.to_string(index=False))
size_df.to_csv(MET_DIR / "dataset_sizes.csv", index=False)
print(f"  ✓ {MET_DIR / 'dataset_sizes.csv'}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Polarity distribution per domain
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/7] Polarity distribution")
pol_df = (df.groupby(["domain", "polarity"])
            .size()
            .reset_index(name="count"))
pol_df.to_csv(MET_DIR / "polarity_distribution.csv", index=False)
print(f"  ✓ {MET_DIR / 'polarity_distribution.csv'}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
for ax, domain in zip(axes, ["laptops", "restaurants"]):
    sub = pol_df[pol_df["domain"] == domain]
    bars = ax.bar(sub["polarity"], sub["count"],
                  color=[PALETTE.get(p, "grey") for p in sub["polarity"]],
                  edgecolor="black", linewidth=0.6)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=10)
    ax.set_title(f"{domain.capitalize()} — Polarity Distribution", fontsize=13)
    ax.set_xlabel("Polarity"); ax.set_ylabel("Count")
plt.tight_layout()
save(fig, "eda_polarity_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Train / test split comparison
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/7] Train vs test polarity balance")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, domain in zip(axes, ["laptops", "restaurants"]):
    sub = df[df["domain"] == domain]
    ct = (sub.groupby(["split", "polarity"])
              .size()
              .unstack(fill_value=0))
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct.plot(kind="bar", ax=ax,
                color=[PALETTE.get(c, "grey") for c in ct_pct.columns],
                edgecolor="black", linewidth=0.6, rot=0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title(f"{domain.capitalize()} — Split × Polarity (%)", fontsize=13)
    ax.set_xlabel("Split"); ax.set_ylabel("Proportion (%)")
    ax.legend(title="Polarity", fontsize=9)
plt.tight_layout()
save(fig, "eda_split_polarity_balance.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Sentence length distribution
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/7] Sentence length distribution")
df["sent_len"] = df["sentence"].str.split().str.len()
len_stats = df.groupby("domain")["sent_len"].describe().round(2)
print(len_stats)
len_stats.to_csv(MET_DIR / "sentence_length_stats.csv")
print(f"  ✓ {MET_DIR / 'sentence_length_stats.csv'}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, domain in zip(axes, ["laptops", "restaurants"]):
    sub = df[df["domain"] == domain]["sent_len"]
    ax.hist(sub, bins=30, color="coral", edgecolor="black", linewidth=0.5)
    ax.axvline(sub.mean(), color="navy",  linestyle="--", label=f"mean={sub.mean():.1f}")
    ax.axvline(sub.median(), color="green", linestyle=":",  label=f"median={sub.median():.1f}")
    ax.set_title(f"{domain.capitalize()} — Sentence Length (words)", fontsize=13)
    ax.set_xlabel("Words"); ax.set_ylabel("Count")
    ax.legend(fontsize=9)
plt.tight_layout()
save(fig, "eda_sentence_length.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Top aspect terms
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5/7] Top aspect terms")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, domain in zip(axes, ["laptops", "restaurants"]):
    sub = df[df["domain"] == domain]["aspect_term"].str.lower()
    top = sub.value_counts().head(15)
    top.plot(kind="barh", ax=ax, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.invert_yaxis()
    ax.set_title(f"{domain.capitalize()} — Top 15 Aspect Terms", fontsize=13)
    ax.set_xlabel("Count")
    # save top-30 to csv
    sub.value_counts().head(30).reset_index().rename(
        columns={"index": "aspect_term", "aspect_term": "count"}
    ).to_csv(MET_DIR / f"top_aspects_{domain}.csv", index=False)
    print(f"  ✓ {MET_DIR / f'top_aspects_{domain}.csv'}")
plt.tight_layout()
save(fig, "eda_top_aspect_terms.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Aspect term polarity heatmap (top-10 aspects per domain)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[6/7] Aspect × polarity heatmap")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, domain in zip(axes, ["laptops", "restaurants"]):
    sub = df[df["domain"] == domain].copy()
    sub["aspect_term"] = sub["aspect_term"].str.lower()
    top10 = sub["aspect_term"].value_counts().head(10).index
    pivot = (sub[sub["aspect_term"].isin(top10)]
             .groupby(["aspect_term", "polarity"])
             .size()
             .unstack(fill_value=0))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax,
                linewidths=0.4, cbar_kws={"shrink": 0.8})
    ax.set_title(f"{domain.capitalize()} — Aspect × Polarity (top-10)", fontsize=13)
    ax.set_ylabel("Aspect Term"); ax.set_xlabel("Polarity")
plt.tight_layout()
save(fig, "eda_aspect_polarity_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Sentences per unique text (multi-aspect sentences)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[7/7] Multi-aspect sentence analysis")
multi = (df.groupby(["domain", "sentence"])
           .size()
           .reset_index(name="aspect_count"))
multi_stats = multi.groupby(["domain", "aspect_count"]).size().reset_index(name="sentences")
multi_stats.to_csv(MET_DIR / "multi_aspect_counts.csv", index=False)
print(f"  ✓ {MET_DIR / 'multi_aspect_counts.csv'}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, domain in zip(axes, ["laptops", "restaurants"]):
    sub = multi[multi["domain"] == domain]["aspect_count"]
    vc = sub.value_counts().sort_index()
    ax.bar(vc.index.astype(str), vc.values, color="mediumpurple",
           edgecolor="black", linewidth=0.5)
    ax.set_title(f"{domain.capitalize()} — Aspects per Sentence", fontsize=13)
    ax.set_xlabel("# Aspects"); ax.set_ylabel("# Sentences")
plt.tight_layout()
save(fig, "eda_multi_aspect_sentences.png")


# ── Final summary ─────────────────────────────────────────────────────────────
print("\n✅ EDA complete!")
print(f"   Charts  → {FIG_DIR}")
print(f"   Metrics → {MET_DIR}")
print("\nFiles saved:")
for f in sorted(FIG_DIR.glob("eda_*.png")):
    print(f"   {f.name}")
for f in sorted(MET_DIR.glob("*.csv")):
    print(f"   {f.name}")