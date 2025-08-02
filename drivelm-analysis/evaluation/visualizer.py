"""
Visualization helpers for evaluation results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class EvaluationVisualizer:
    def __init__(self, results_df: pd.DataFrame):
        self.results_df = results_df

    @staticmethod
    def plot_similarity_distribution(per_sample: list):

        if not per_sample:
            print("⚠️ No per-sample results to visualize")
            return None

        df = pd.DataFrame(per_sample)
        if "semantic_sim" not in df.columns:
            print("⚠️ No 'semantic_sim' field found in results")
            return None

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df["semantic_sim"], bins=20, kde=True, color="blue", ax=ax)
        ax.set_title("Semantic Similarity Distribution")
        ax.set_xlabel("Semantic Similarity Score")
        ax.set_ylabel("Frequency")
        ax.grid(True, linestyle="--", alpha=0.6)
        return fig

    def plot_confidence_distribution(self):
        if "confidence" not in self.results_df.columns:
            print("⚠️ No 'confidence' column in results")
            return None

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(self.results_df["confidence"], bins=20, kde=True, ax=ax)
        ax.set_title("Confidence Distribution")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Frequency")
        return fig

    @staticmethod
    def plot_metrics_bar(metrics: dict):
        fig, ax = plt.subplots(figsize=(8, 5))
        names, values = list(metrics.keys()), list(metrics.values())
        sns.barplot(x=names, y=values, ax=ax)
        ax.set_title("Evaluation Metrics")
        ax.set_ylim(0, 1)
        return fig

    @staticmethod
    def plot_quality_distribution(per_sample: list):
      
        if not per_sample:
            return None

        df = pd.DataFrame(per_sample)
        if "semantic_sim" not in df.columns:
            return None

        # Categorize quality levels
        df["quality"] = pd.cut(
            df["semantic_sim"],
            bins=[-float("inf"), 0.4, 0.7, float("inf")],
            labels=["Poor", "Medium", "Good"]
        )

        quality_counts = df["quality"].value_counts().reindex(["Good", "Medium", "Poor"], fill_value=0)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=quality_counts.index, y=quality_counts.values, ax=ax, palette="coolwarm")
        ax.set_title("Answer Quality Distribution")
        ax.set_xlabel("Quality Category")
        ax.set_ylabel("Number of Samples")
        return fig

    def show_examples(self, n: int = 5):
        print("📋 Sample Predictions vs References:\n")
        sample = self.results_df.sample(min(n, len(self.results_df)))
        for _, row in sample.iterrows():
            print(f"Q: {row['question']}")
            print(f"Reference: {row.get('reference', '')}")
            print(f"Prediction: {row.get('prediction', '')}")
            if "confidence" in row:
                print(f"Confidence: {row['confidence']:.2f}")
            print("-" * 60)
