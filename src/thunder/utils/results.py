import os

METRICS = ["accuracy", "balanced_accuracy", "f1", "jaccard", "roc_auc"]

TASK2CATS = {
    "adversarial_attack": ["adversarial", "clean"],
    "simple_shot": ["1", "2", "4", "8", "16"],
    "image_retrieval": ["1", "3", "5", "10"],
}


def extract_value_from_large_json(
    file_path: str, target_keys: list, categories: list = None
) -> dict:
    """
    Extracts values from a large JSON file using ijson.
    :param file_path: Path to the JSON file.
    :param target_keys: List of keys to extract values for.
    :param categories: List of first-level keys (each pointing to a dict)

    :return: Dictionary with keys and their corresponding values."""
    import ijson

    if categories is None:
        values_dict = {key: None for key in target_keys}
    else:
        values_dict = {cat: {key: None for key in target_keys} for cat in categories}

    with open(file_path, "rb") as f:  # Open the file in binary mode
        parser = ijson.parse(f)
        current_key = None
        if categories is not None:
            current_cat = None
        for _, event, value in parser:
            if categories is None:
                if event == "map_key" and value in target_keys:
                    current_key = value
                elif current_key is not None and event in [
                    "string",
                    "number",
                    "boolean",
                    "null",
                ]:
                    values_dict[current_key] = value
                    current_key = None
                if all([val is not None for val in values_dict.values()]):
                    return values_dict
            else:
                if event == "map_key" and value in categories:
                    current_cat = value
                elif event == "map_key" and value in target_keys:
                    current_key = value
                elif current_key is not None and event in [
                    "string",
                    "number",
                    "boolean",
                    "null",
                ]:
                    values_dict[current_cat][current_key] = value
                    current_key = None

    return values_dict


def gather_results():
    import glob
    import pandas as pd

    print("Gathering results...")
    results_dir = os.path.join(os.environ["THUNDER_BASE_DATA_FOLDER"], "outputs", "res")
    results_files = glob.glob(f"{results_dir}/**/*.json", recursive=True)

    res = []
    for file in results_files:
        dataset, model, task, adaptation, _ = file.split("res/")[1].split("/")

        # Init base dict
        base_dict = {
            "dataset": dataset,
            "model": model,
            "task": task,
            "adaptation": adaptation,
        }

        # Extracting results
        if task in TASK2CATS.keys():
            categories = TASK2CATS[task]
            result_dicts = extract_value_from_large_json(file, METRICS, categories)
        else:
            result_dicts = {
                "": extract_value_from_large_json(file, METRICS),
            }

        # Populating rows of results
        for setting, result_dict in result_dicts.items():
            for metric in METRICS:
                metric_dict = base_dict.copy()
                metric_dict["metric"] = metric
                value = result_dict[metric]
                if value is not None:
                    value = round(100 * value, 1)
                metric_dict["value"] = value
                metric_dict["setting"] = setting
                res.append(metric_dict)

    df = pd.DataFrame(res)
    df.to_csv(os.path.join(results_dir, "results.csv"), index=False)
    print("Saved at:", os.path.join(results_dir, "results.csv"))
    print(
        "The setting column corresponds to: (i) Adversarial/Clean for adversarial_attack, (ii) nb shots for simple_shot, (iii) k for image_retrieval."
    )
