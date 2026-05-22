import os

import pandas as pd
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from zoology.analysis.utils import fetch_wandb_runs

MODEL_NAME = "name"
DMODEL_NAME = "d_model"


model2color = {
    "Hyena": "#BAB0AC",
    "H3": "#B07AA1",
    
    'Attention': "black", 
    'Sliding window attention': "black", 

    'Based': "#59A14F", # "#F28E2B"
    "Mamba2": "#4E79A7",
    'Gated delta net': "#9C755F", 
    'Rwkv7': "#EDC948", 
    'Mamba': "#76B7B2", 
    'Delta net': "#E15759", 
    'Gla': "#F28E2B",

    'Ttt linear': "#D3722C",
    'Ttt mlp': "#EDC948",
    'CSA': "#AF7AA1", # 一种紫色
    'Deepseek csa hca': "#8CD17D", # 一种绿色
    'New Model 3': "#FF9DA7", # 一种粉色/珊瑚色
}

graph_order = [
    "Attention",
    "Sliding Window",
    "H3",
    "Hyena",
    "Based",
    "Mamba",
    "GLA",
    "Mamba-2",
    "DeltaNet",
    "RWKV-7",
    "Gated DeltaNet",
    'TTT Linear',
    'TTT MLP',
    'CSA',
    'Deepseek csa hca',
]

name_replacements = {
    "Mamba2": "Mamba-2",
    "Gla": "GLA",
    "Rwkv7": "RWKV-7",
    "Mamba": "Mamba",
    "Delta net": "DeltaNet",
    "Gated delta net": "Gated DeltaNet",
    "Sliding window attention": "Sliding Window",
    "Ttt linear": "TTT Linear",
    "Ttt mlp": "TTT MLP",
    'csa':'CSA',
    'deepseek-csa-hca':'Deepseek csa hca',
}

def _normalize_model_key(s: str) -> str:
    """
    Normalize raw model names to match keys in name_replacements.
    We preserve hyphens as given in the raw name (e.g., Mamba2 -> Mamba2),
    but convert underscores to spaces and standardize case for lookup.
    """
    if not isinstance(s, str):
        return s
    s2 = s.strip().replace("_", " ")
    # Don't force hyphen removal—our replacements decide final punctuation.
    # Use case-insensitive lookup by lowercasing.
    return s2

def _apply_name_replacements(series: pd.Series) -> pd.Series:
    # case-insensitive map using provided name_replacements keys
    lower_map = {k.lower(): v for k, v in name_replacements.items()}
    return series.apply(lambda x: lower_map.get(_normalize_model_key(x).lower(), _normalize_model_key(x)))

def _mapped_palette(base_palette: dict, replacements: dict) -> dict:
    """
    Map your existing model2color keys through name_replacements so
    colors line up with the final display names.
    """
    lower_map = {k.lower(): v for k, v in replacements.items()}
    mapped = {}
    for k, color in base_palette.items():
        key_norm = _normalize_model_key(k)
        final_name = lower_map.get(key_norm.lower(), key_norm)
        mapped[final_name] = color
    return mapped


def plot(
    df: pd.DataFrame,
    metric: str="valid/accuracy",
):

    idx = df.groupby(
        ["state_size", MODEL_NAME]
    )[metric].idxmax(skipna=True).dropna()
    plot_df = df.loc[idx]

    plot_df[DMODEL_NAME] = plot_df[MODEL_NAME].str.extract(r'd(\d+)')

    # upper case the model names first letter
    plot_df[MODEL_NAME] = plot_df[MODEL_NAME].str.capitalize()
    # replace "-" and "_" with " "
    plot_df[MODEL_NAME] = plot_df[MODEL_NAME].str.replace("-", " ")
    plot_df[MODEL_NAME] = plot_df[MODEL_NAME].str.replace("_", " ")
    # replace model column name with "Model"
    plot_df["Model"] = _apply_name_replacements(plot_df[MODEL_NAME])

    # # (06/05) adjust the state sizes for rwkv v7
    # rwkv_mask = (plot_df["Model"] == "Rwkv7")
    # rwkv_mask_128 = (plot_df["Model"] == "Rwkv7") & (plot_df[DMODEL_NAME] == 128)
    # rwkv_mask_256 = (plot_df["Model"] == "Rwkv7") & (plot_df[DMODEL_NAME] == 256)
    # print(plot_df[['Model', 'state_size', DMODEL_NAME]][rwkv_mask_128 | rwkv_mask_256])
    # plot_df.loc[rwkv_mask_128, "state_size"] /= 4
    # plot_df.loc[rwkv_mask_256, "state_size"] /= 16
    # print(plot_df[['Model', 'state_size', DMODEL_NAME]][rwkv_mask])

    palette = _mapped_palette(model2color, name_replacements)

    # exclude models in list
    list_to_exclude = ["Mamba", "RWKV-7", "GLA"]
    plot_df = plot_df[~plot_df["Model"].isin(list_to_exclude)]
    filtered_order = [o for o in graph_order if o not in list_to_exclude]
    palette = {k: v for k, v in palette.items() if k not in list_to_exclude}

    # sns.set_theme(style="whitegrid")
    g = sns.relplot(
        data=plot_df,
        y=metric,
        x="state_size",
        hue="Model",
        kind="scatter",
        marker="o",
        hue_order=filtered_order,           # enforce your order
        height=5,
        aspect=1,
        palette=palette,
        s=60,
        edgecolor="black",    # <-- thin black border
        linewidth=0.5,        # <-- thickness of the border
    )
    g.set(xscale="log", ylabel="Recall Accuracy", xlabel="State Size (log scale)")

    ax = g.ax
    ax.set_xlabel("State Size (log scale)", fontsize=16)
    ax.set_ylabel("Recall Accuracy", fontsize=16)
    if g._legend is not None:
        g._legend.set_title("Model", prop={"size": 16})
    #title
    ax.set_title("MQAR (Standard)", fontsize=16)


    ax = g.ax

    # --- Find the leftmost point (smallest state_size) with accuracy near 1.0 ---
    # tweak tolerance if needed (here: >= 0.99 accuracy)
    filtered_sorted_df = plot_df.loc[plot_df[metric] >= 0.99, ["state_size", metric]].sort_values("state_size")

    first_idx = filtered_sorted_df.index[0] if not filtered_sorted_df.empty else None

    if first_idx is None:
        print(f"No data points found where {metric} >= 0.99")
        return

    point = filtered_sorted_df.loc[first_idx]
    x_val, y_val = point["state_size"], point[metric]

    # --- Draw vertical dashed line ---
    ax.axvline(
        x=x_val,
        ymin=0, ymax=y_val,  # scale 0–1 relative to axis
        linestyle="--",
        color="black",
        linewidth=1,
    )



# You can find the "default-2025..." tags in the wandb UI under the "launch_id" key for a run. 
# Each sweep you launch (with an experiments config file) will have a shared launch_id. 
# NOTE: The "project_name" is the name of the wandb project!
if __name__ == "__main__" :
    df = fetch_wandb_runs(
        launch_id=[
            "default-2026-05-25-09-17-08",
        ], 
        project_name="zoology-deepseek_csa-hca"
    )

    # df2 = fetch_wandb_runs(
    #     launch_id=[
    #         # Adding RWKV-v7
    #         "default-2025-03-04-16-43-26",
    #         "default-2025-03-04-15-55-12",
    #         "default-2025-03-04-15-11-23"

    #         # Adding NSA
    #         "default-2025-03-06-11-46-30",

    #         # Adding DeltaNet
    #         "default-2025-03-05-14-30-11",
    #         "default-2025-03-05-14-07-18",
    #         "default-2025-03-05-14-59-58",

    #         # Adding Gated DeltaNet
    #         "default-2025-03-05-16-20-42",
    #         "default-2025-03-05-16-41-32",

    #         # Adding Gated Linear Attention (GLA)
    #         "default-2025-03-05-16-01-15",

    #         # Adding TTT Linear
    #         "default-2026-01-09-20-46-16",
    #         # Adding TTT MLP
    #         "default-2026-01-09-22-37-36",
    #     ], 
    #     project_name="0325_zoology"
    # )

    # # add the new runs to the df
    # df = pd.concat([df, df2]).reset_index(drop=True)

    plot(df=df)

    # save in high resolution
    plt.savefig("results.png", dpi=300, bbox_inches="tight")
    print("results.png")


