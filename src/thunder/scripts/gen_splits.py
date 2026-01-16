from src.thunder import benchmark
from tqdm.auto import tqdm
from src.thunder.datasets.download import download_datasets
from src.thunder.datasets.data_splits import generate_splits

if __name__ == "__main__":
    datasets = [
        "bach",
        "bracs",
        "break_his",
        "ccrcc",
        "crc",
        "esca",
        # "mhist",
        "ocelot",
        "pannuke",
        "patch_camelyon",
        # "segpath_epithelial",
        # "segpath_lymphocytes",
        "tcga_crc_msi",
        "tcga_tils",
        "tcga_uniform",
        # "wilds",
    ]
    generate_splits(datasets)
