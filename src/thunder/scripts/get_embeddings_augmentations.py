from src.thunder import benchmark
from tqdm.auto import tqdm
import itertools

if __name__ == "__main__":
    task = "linear_probing"
    datasets = [
        # "bach",
        # "bracs",
        # "break_his",
        # "ccrcc",
        "crc",
        "esca",
        # "mhist",
        # "ocelot",
        # "pannuke",
        # "patch_camelyon",
        # "segpath_epithelial",
        # "segpath_lymphocytes",
        # "tcga_crc_msi",
        "tcga_tils",
        # "tcga_uniform",
        # "wilds",
    ]
    models = [
        "uni",
        "uni2h",
        # "virchow",
        # "virchow2",
        # "hoptimus0",
        # "hoptimus1",
        "conch",
        # "titan",
        # "phikon",
        # "phikon2",
        # "hiboub",
        # "hiboul",
        # "midnight",
        # "keep",
        # "quiltb32",
        # "plip",
        # "musk",
        # "dinov2base",
        # "dinov2large",
        # "vitbasepatch16224in21k",
        # "vitlargepatch16224in21k",
        "clipvitbasepatch32",
        # "clipvitlargepatch14",
    ]
    for dataset, model in tqdm(itertools.product(datasets, models)):
        benchmark(model, dataset, "pre_computing_augmentation_embeddings")
