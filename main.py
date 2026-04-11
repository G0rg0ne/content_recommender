"""
MovieLens 25M — Exploratory Data Analysis
==========================================
Download data from: https://grouplens.org/datasets/movielens/25m/
Extract the zip, then set DATA_DIR below to where the CSVs live.

Run:  python movielens_eda.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from scipy.sparse import csr_matrix

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR  = "data/ml-25m"          # folder containing ratings.csv, movies.csv, tags.csv
OUTPUT_DIR = "data/eda_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "primary":   "#534AB7",
    "secondary": "#1D9E75",
    "accent":    "#D85A30",
    "muted":     "#888780",
    "light":     "#EEEDFE",
}


# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data...")

ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv",
                      dtype={"userId": "int32", "movieId": "int32",
                             "rating": "float32", "timestamp": "int64"})

movies  = pd.read_csv(f"{DATA_DIR}/movies.csv",
                      dtype={"movieId": "int32", "title": "str", "genres": "str"})

tags    = pd.read_csv(f"{DATA_DIR}/tags.csv",
                      dtype={"userId": "int32", "movieId": "int32",
                             "tag": "str", "timestamp": "int64"})

ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")

print(f"  Ratings : {len(ratings):>12,}")
print(f"  Users   : {ratings['userId'].nunique():>12,}")
print(f"  Movies  : {ratings['movieId'].nunique():>12,}")
print(f"  Tags    : {len(tags):>12,}")
print(f"  Date range: {ratings['datetime'].min().date()} → {ratings['datetime'].max().date()}")


# ── 2. Basic stats ────────────────────────────────────────────────────────────
print("\n── Rating value distribution ──")
print(ratings["rating"].describe().to_string())

n_users  = ratings["userId"].nunique()
n_items  = ratings["movieId"].nunique()
sparsity = 1 - len(ratings) / (n_users * n_items)
print(f"\nMatrix sparsity: {sparsity:.4%}  ({n_users} users × {n_items} items)")


# ── 3. Per-user & per-item stats ──────────────────────────────────────────────
user_stats = ratings.groupby("userId").agg(
    n_ratings   = ("rating", "count"),
    mean_rating = ("rating", "mean"),
    std_rating  = ("rating", "std"),
).reset_index()

item_stats = ratings.groupby("movieId").agg(
    n_ratings   = ("rating", "count"),
    mean_rating = ("rating", "mean"),
).reset_index()

print("\n── Users ──")
print(user_stats["n_ratings"].describe().astype(int).to_string())
print("\n── Items ──")
print(item_stats["n_ratings"].describe().astype(int).to_string())


# ── 4. Genre explosion ────────────────────────────────────────────────────────
genres_df = (
    movies[movies["genres"] != "(no genres listed)"]
    .assign(genre=lambda d: d["genres"].str.split("|"))
    .explode("genre")
)
genre_counts = genres_df["genre"].value_counts()


# ── 5. Extract year from title ────────────────────────────────────────────────
movies["year"] = (
    movies["title"]
    .str.extract(r"\((\d{4})\)$", expand=False)
    .astype("float")
)
movies_with_year = movies.dropna(subset=["year"])
movies_with_year = movies_with_year[
    movies_with_year["year"].between(1900, 2025)
]


# ── 6. Temporal activity ──────────────────────────────────────────────────────
ratings["year_month"] = ratings["datetime"].dt.to_period("M")
monthly_activity = ratings.groupby("year_month").size().reset_index(name="count")
monthly_activity["dt"] = monthly_activity["year_month"].dt.to_timestamp()


# ── 7. Cold-start zones ───────────────────────────────────────────────────────
cold_user_pct  = (user_stats["n_ratings"] < 5).mean() * 100
cold_item_pct  = (item_stats["n_ratings"] < 5).mean() * 100
power_user_pct = (user_stats["n_ratings"] > 200).mean() * 100

print(f"\n── Cold-start analysis ──")
print(f"  Users with < 5 ratings  : {cold_user_pct:.1f}%")
print(f"  Items with < 5 ratings  : {cold_item_pct:.1f}%")
print(f"  Power users (>200 rtgs) : {power_user_pct:.1f}%")


# ── 8. Plots ──────────────────────────────────────────────────────────────────
print("\nGenerating plots...")

# ── 8a. Rating distribution ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
vals, cnts = np.unique(ratings["rating"], return_counts=True)
bars = ax.bar(vals, cnts / 1e6, width=0.4, color=COLORS["primary"], alpha=0.85)
ax.set_xlabel("Rating value", fontsize=12)
ax.set_ylabel("Count (millions)", fontsize=12)
ax.set_title("Rating value distribution", fontsize=14, fontweight="bold")
ax.set_xticks(vals)
for bar, c in zip(bars, cnts):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05, f"{c/1e6:.1f}M",
            ha="center", va="bottom", fontsize=9, color=COLORS["muted"])
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_rating_distribution.png", dpi=150)
plt.show()
print("  Saved 01_rating_distribution.png")

# ── 8b. User activity (log-scale histogram) ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(user_stats["n_ratings"], bins=80,
             color=COLORS["primary"], alpha=0.8, edgecolor="none")
axes[0].set_yscale("log")
axes[0].set_xlabel("Ratings per user", fontsize=11)
axes[0].set_ylabel("Number of users (log)", fontsize=11)
axes[0].set_title("User activity distribution", fontsize=13, fontweight="bold")
axes[0].axvline(5,   color=COLORS["accent"],    linestyle="--", linewidth=1.5, label="cold-start (<5)")
axes[0].axvline(200, color=COLORS["secondary"], linestyle="--", linewidth=1.5, label="power user (>200)")
axes[0].legend(fontsize=9)

axes[1].hist(item_stats["n_ratings"], bins=80,
             color=COLORS["secondary"], alpha=0.8, edgecolor="none")
axes[1].set_yscale("log")
axes[1].set_xlabel("Ratings per item", fontsize=11)
axes[1].set_ylabel("Number of items (log)", fontsize=11)
axes[1].set_title("Item popularity distribution (long tail)", fontsize=13, fontweight="bold")
axes[1].axvline(5, color=COLORS["accent"], linestyle="--", linewidth=1.5, label="cold-start (<5)")
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_activity_distributions.png", dpi=150)
plt.show()
print("  Saved 02_activity_distributions.png")

# ── 8c. Long tail: cumulative coverage ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
sorted_items = item_stats.sort_values("n_ratings", ascending=False).reset_index(drop=True)
sorted_items["cum_pct_ratings"] = sorted_items["n_ratings"].cumsum() / sorted_items["n_ratings"].sum() * 100
sorted_items["item_pct"] = (sorted_items.index + 1) / len(sorted_items) * 100

ax.plot(sorted_items["item_pct"], sorted_items["cum_pct_ratings"],
        color=COLORS["primary"], linewidth=2)
ax.fill_between(sorted_items["item_pct"], sorted_items["cum_pct_ratings"],
                alpha=0.1, color=COLORS["primary"])

for pct in [20, 50]:
    idx = (sorted_items["item_pct"] - pct).abs().idxmin()
    cov = sorted_items.loc[idx, "cum_pct_ratings"]
    ax.axvline(pct, color=COLORS["muted"], linestyle=":", linewidth=1)
    ax.axhline(cov, color=COLORS["muted"], linestyle=":", linewidth=1)
    ax.annotate(f"Top {pct}% items\n→ {cov:.0f}% of ratings",
                xy=(pct, cov), xytext=(pct + 3, cov - 12),
                fontsize=9, color=COLORS["muted"],
                arrowprops=dict(arrowstyle="->", color=COLORS["muted"], lw=0.8))

ax.set_xlabel("Item percentile (most → least popular)", fontsize=11)
ax.set_ylabel("Cumulative % of all ratings", fontsize=11)
ax.set_title("Long tail: item popularity coverage", fontsize=13, fontweight="bold")
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_long_tail.png", dpi=150)
plt.show()
print("  Saved 03_long_tail.png")

# ── 8d. Genre distribution ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
genre_counts_plot = genre_counts.sort_values()
colors_bar = [COLORS["primary"] if i >= len(genre_counts_plot) - 5
              else COLORS["light"] for i in range(len(genre_counts_plot))]
# use a blue-ish palette for contrast
import matplotlib.cm as cm
cmap_vals = cm.get_cmap("Blues")(
    np.linspace(0.3, 0.9, len(genre_counts_plot))
)
hbars = ax.barh(genre_counts_plot.index, genre_counts_plot.values / 1e3,
                color=cmap_vals, edgecolor="none")
ax.set_xlabel("Number of movies (thousands)", fontsize=11)
ax.set_title("Movie count by genre", fontsize=13, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}k"))
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_genre_distribution.png", dpi=150)
plt.show()
print("  Saved 04_genre_distribution.png")

# ── 8e. Release year distribution ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
year_counts = movies_with_year["year"].value_counts().sort_index()
ax.bar(year_counts.index, year_counts.values,
       width=0.9, color=COLORS["secondary"], alpha=0.8, edgecolor="none")
ax.set_xlabel("Release year", fontsize=11)
ax.set_ylabel("Number of movies", fontsize=11)
ax.set_title("Movies by release year", fontsize=13, fontweight="bold")
ax.set_xlim(1900, 2025)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_release_years.png", dpi=150)
plt.show()
print("  Saved 05_release_years.png")

# ── 8f. Monthly rating activity ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
ax.fill_between(monthly_activity["dt"], monthly_activity["count"] / 1e6,
                alpha=0.3, color=COLORS["primary"])
ax.plot(monthly_activity["dt"], monthly_activity["count"] / 1e6,
        color=COLORS["primary"], linewidth=1.5)
ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Ratings per month (millions)", fontsize=11)
ax.set_title("Rating activity over time", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_temporal_activity.png", dpi=150)
plt.show()
print("  Saved 06_temporal_activity.png")

# ── 8g. Mean rating per user vs activity ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
sample = user_stats.sample(min(5000, len(user_stats)), random_state=42)
sc = ax.scatter(sample["n_ratings"], sample["mean_rating"],
                alpha=0.25, s=8, c=COLORS["primary"])
ax.set_xscale("log")
ax.set_xlabel("Number of ratings (log scale)", fontsize=11)
ax.set_ylabel("Mean rating given by user", fontsize=11)
ax.set_title("User activity vs. mean rating (sample of 5k users)", fontsize=13, fontweight="bold")
ax.axhline(ratings["rating"].mean(), color=COLORS["accent"],
           linestyle="--", linewidth=1.2, label=f"Global mean = {ratings['rating'].mean():.2f}")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_user_activity_vs_rating.png", dpi=150)
plt.show()
print("  Saved 07_user_activity_vs_rating.png")

# ── 8h. Sparsity visualisation (sample submatrix) ─────────────────────────────
print("\nBuilding sample sparsity matrix...")
top_users = user_stats.nlargest(200, "n_ratings")["userId"].values
top_items = item_stats.nlargest(300, "n_ratings")["movieId"].values

sub = ratings[
    ratings["userId"].isin(top_users) &
    ratings["movieId"].isin(top_items)
].copy()

uid_map  = {u: i for i, u in enumerate(top_users)}
iid_map  = {m: i for i, m in enumerate(top_items)}
sub["u"] = sub["userId"].map(uid_map)
sub["i"] = sub["movieId"].map(iid_map)

mat = csr_matrix(
    (sub["rating"].values, (sub["u"].values, sub["i"].values)),
    shape=(200, 300)
).toarray()
mat_display = np.where(mat > 0, 1.0, np.nan)

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(mat_display, aspect="auto", cmap="Blues",
               interpolation="none", vmin=0, vmax=1)
ax.set_xlabel("Item index (top 300 most-rated)", fontsize=11)
ax.set_ylabel("User index (top 200 most-active)", fontsize=11)
ax.set_title(
    f"Interaction matrix sample — {(mat > 0).sum() / mat.size:.1%} dense\n"
    f"(Full matrix: {sparsity:.3%} sparse)",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_sparsity_matrix.png", dpi=150)
plt.show()
print("  Saved 08_sparsity_matrix.png")


# ── 9. Summary printout ───────────────────────────────────────────────────────
print("\n" + "="*55)
print("  DATASET SUMMARY")
print("="*55)
print(f"  Total ratings         : {len(ratings):>12,}")
print(f"  Unique users          : {n_users:>12,}")
print(f"  Unique movies         : {n_items:>12,}")
print(f"  Matrix sparsity       : {sparsity:>11.3%}")
print(f"  Global mean rating    : {ratings['rating'].mean():>12.3f}")
print(f"  Median ratings/user   : {user_stats['n_ratings'].median():>12.0f}")
print(f"  Median ratings/item   : {item_stats['n_ratings'].median():>12.0f}")
print(f"  Cold users (<5 rtgs)  : {cold_user_pct:>11.1f}%")
print(f"  Cold items (<5 rtgs)  : {cold_item_pct:>11.1f}%")
print(f"  Power users (>200)    : {power_user_pct:>11.1f}%")
print(f"  Most popular genre    : {genre_counts.index[0]}")
print(f"  Date range            : {ratings['datetime'].min().date()} → {ratings['datetime'].max().date()}")
print("="*55)
print(f"\nAll plots saved to ./{OUTPUT_DIR}/")
print("Done.")