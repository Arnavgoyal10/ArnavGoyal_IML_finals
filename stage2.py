import os
from Bio import SeqIO
import multiprocessing


def process_folder(folder_path, window_size=150, final_path="output"):
    print(folder_path)
    temp = folder_path.split("/")[1]
    # Define paths to the input files within the given folder
    input_files = [
        os.path.join(folder_path, f"output.amino_{i}.fasta") for i in range(3)
    ]

    # Define the output path for the combined file in the same folder
    output_file = os.path.join(final_path, f"{temp}/output_final.fasta")

    with open(output_file, "w") as output_handle:
        sequenceno = 0  # Initialize the sequence number across all files in the folder

        # Loop over each input file
        for input_file in input_files:
            for record in SeqIO.parse(input_file, "fasta"):
                sequence = str(record.seq)
                seq_length = len(sequence)
                end = []
                end.append(0)
                entry_count = 1

                # Iterate over the sequence in steps of window_size
                for i in range(0, seq_length):
                    if sequence[i] == "*" and sequence[i - 1] != "*":
                        end.append(i)

                for i in range(1, len(end)):
                    subsequence = sequence[end[i - 1] : end[i]]
                    # Add filters for sequence length and all underscore sequences
                    print("condition_check")
                    if len(subsequence) >= 10 and all(
                        char != "_" for char in subsequence
                    ):
                        new_id = f"{record.id}.{entry_count}.{sequenceno}.{end[i-1]}.{end[i]}"
                        output_handle.write(f">{new_id}\n")
                        output_handle.write(f"{subsequence}\n")
                        entry_count += 1

            sequenceno += 1


def main():
    # Define the list of folders to process
    folder_paths = [
        # "data1/Drosophila_melanogaster",
        # "data1/eastern_oyster",
        "data1/Human",
        # "data1/lion",
        # "data1/zebra_fish",
    ]

    # Create a pool of worker processes
    with multiprocessing.Pool() as pool:
        # Map each folder to the process_folder function using multiprocessing
        pool.map(process_folder, folder_paths)


if __name__ == "__main__":
    main()
