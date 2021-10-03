import matplotlib.pyplot as plt
from pathlib import Path


class global_config:
    """This is a Global Config Class."""

    # Directories
    BASE_DIR = Path(__file__).parent.parent.absolute()
    DATA_DIR = Path(BASE_DIR, "data")

    # File Path using Absolute Path to ease user's experience
    raw_data = Path(DATA_DIR, "raw/data.csv")
    processed_final = Path(DATA_DIR, "processed/processed.csv")
    spot_checking = Path(DATA_DIR, "results/spot_checking.csv")
    spot_checking_summary = Path(DATA_DIR, "results/spot_checking_summary.csv")
    spot_checking_boxplot = Path(DATA_DIR, "images/spot_checking_boxplot.png")
    oof_confusion_matrix = Path(DATA_DIR, "images/oof_confusion_matrix.png")
    final_train_confusion_matrix = Path(DATA_DIR, "images/final_train_confusion_matrix.png")
    precision_recall_threshold_plot = Path(DATA_DIR, "images/precision_recall_threshold_plot.png")
    roc_plot = Path(DATA_DIR, "images/roc_plot.png")

    # Data Information
    target = ["diagnosis"]
    unwanted_cols = ["id", "Unnamed: 32"]
    class_dict = {"B": 0, "M": 1}

    # Plotting
    colors = ["#fe4a49", "#2ab7ca", "#fed766", "#59981A"]
    cmap_reversed = plt.cm.get_cmap("mako_r")

    # Seed Number
    seed = 1992

    # Cross Validation
    num_folds = 5
    cv_schema = "StratifiedKFold"
    split_size = {"train_size": 0.9, "test_size": 0.1}
