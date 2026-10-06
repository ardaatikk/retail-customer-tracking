from src.analytics.heatmap import generate_heatmap
from src.analytics.summary import generate_summary
from src.analytics.trajectory import generate_trajectory


def run_analytics_pipeline() -> None:
    print()
    print("Retail analytics pipeline")
    print("=========================")
    print()

    print("[1/3] Generating trajectory...")
    generate_trajectory()

    print()

    print("[2/3] Generating heatmap...")
    generate_heatmap()

    print()

    print("[3/3] Generating customer & zone summary...")
    generate_summary()

    print()

    print("Analytics pipeline complete")
    print("===========================")


if __name__ == "__main__":
    run_analytics_pipeline()