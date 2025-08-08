"""
Bu script GTSRB veri setindeki GT-*.csv dosyalarını okuyarak
tek bir annotations.csv dosyası oluşturur.
"""

import os
import pandas as pd
import glob


def create_annotations_csv():

    class_dirs = glob.glob("data/*/")
    class_dirs.sort()

    all_annotations = []

    for class_dir in class_dirs:

        class_id = int(os.path.basename(class_dir.rstrip("/")))

        gt_file = os.path.join(class_dir, f"GT-{class_id:05d}.csv")

        if os.path.exists(gt_file):

            try:
                df = pd.read_csv(gt_file, sep=";")

                df["Filename"] = df["Filename"].apply(
                    lambda x: os.path.join(f"{class_id:05d}", x)
                )

                if "ClassId" not in df.columns:
                    df["ClassId"] = class_id

                all_annotations.append(df)

            except Exception as e:
                print(f"Error reading {gt_file}: {e}")
                continue
        else:
            print(f"GT file not found: {gt_file}")

    if all_annotations:

        final_df = pd.concat(all_annotations, ignore_index=True)

        required_columns = [
            "Filename",
            "Roi.x1",
            "Roi.y1",
            "Roi.x2",
            "Roi.y2",
            "ClassId",
        ]
        missing_columns = set(required_columns) - set(final_df.columns)

        if missing_columns:

            if "Roi.x1" not in final_df.columns:
                final_df["Roi.x1"] = 0
                final_df["Roi.y1"] = 0
                final_df["Roi.x2"] = 128
                final_df["Roi.y2"] = 128

        output_file = "annotations.csv"
        final_df.to_csv(output_file, sep=";", index=False)

        return True
    else:
        return False


if __name__ == "__main__":
    create_annotations_csv()
