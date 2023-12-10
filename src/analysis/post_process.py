import csv
import gzip
import os


def merge_files(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if not filename.endswith(".gz"):
            continue

        with gzip.open(os.path.join(input_dir, filename), 'rt') as f_in:
            reader = csv.DictReader(f_in)

            for row in reader:
                # Extract key from 4th column
                key = row["chrom"]

                # Create output file if it doesn't exist
                output_path = os.path.join(output_dir, f"DITTO_{key}.tsv")
                if not os.path.exists(output_path):
                    with open(output_path, "w") as f_out:
                        writer = csv.DictWriter(f_out, fieldnames=['chrom','pos','ref_base','alt_base','transcript','gene','consequence','DITTO'], delimiter="\t")
                        writer.writeheader()

                # Write row to corresponding output file
                with open(output_path, "a") as f_out:
                    writer = csv.DictWriter(f_out, fieldnames=['chrom','pos','ref_base','alt_base','transcript','gene','consequence','DITTO'], delimiter="\t")
                    writer.writerow(row)


if __name__ == "__main__":
    input_dir = "/data/project/worthey_lab/projects/experimental_pipelines/tarun/DITTO/data/processed/snvs"  # Modify this path
    output_dir = "/data/project/worthey_lab/projects/experimental_pipelines/tarun/DITTO/data/processed/all_snv/"  # Modify this path
    merge_files(input_dir, output_dir)
