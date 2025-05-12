from Bio import SeqIO
from Bio.Seq import Seq
import multiprocessing

genetic_code = {
    "UUU": "F",
    "UUC": "F",
    "UUA": "L",
    "UUG": "L",
    "CUU": "L",
    "CUC": "L",
    "CUA": "L",
    "CUG": "L",
    "AUU": "I",
    "AUC": "I",
    "AUA": "I",
    "AUG": "M",
    "GUU": "V",
    "GUC": "V",
    "GUA": "V",
    "GUG": "V",
    "UCU": "S",
    "UCC": "S",
    "UCA": "S",
    "UCG": "S",
    "CCU": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "ACU": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "GCU": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "UAU": "Y",
    "UAC": "Y",
    "UAA": "*",
    "UAG": "*",
    "CAU": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "AAU": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "GAU": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "UGU": "C",
    "UGC": "C",
    "UGA": "*",
    "UGG": "W",
    "CGU": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AGU": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GGU": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}


def translate_fasta_with_start(input_fasta, path):

    def clean_sequence(sequence, frame):
        # Shift the sequence to start from the given frame
        shifted_sequence = sequence[frame:]
        # Truncate sequence to a multiple of 3
        codon_length = len(shifted_sequence) - (len(shifted_sequence) % 3)

        return shifted_sequence[:codon_length]

    for record in SeqIO.parse(input_fasta, "fasta"):

        translated_record_0 = record
        translated_record_1 = record
        translated_record_2 = record

        clean_sequence0 = clean_sequence(record.seq, frame=0)
        clean_sequence1 = clean_sequence(record.seq, frame=1)
        clean_sequence2 = clean_sequence(record.seq, frame=2)

        rna_sequence0 = clean_sequence0.upper().replace("T", "U")
        rna_sequence1 = clean_sequence1.upper().replace("T", "U")
        rna_sequence2 = clean_sequence2.upper().replace("T", "U")

        rna_sequences = {
            "rna_sequence0": rna_sequence0,
            "rna_sequence1": rna_sequence1,
            "rna_sequence2": rna_sequence2,
        }

        # Function to translate RNA to protein
        def translate_rna_to_protein(rna_sequence):
            rna_sequence = rna_sequence.replace("\n", "")  # Clean up newlines
            protein_sequence = []
            for i in range(0, len(rna_sequence) - 2, 3):
                codon = rna_sequence[i : i + 3]
                protein_sequence.append(
                    genetic_code.get(codon, "_")
                )  # Use "_" for unknown codons
            return "".join(protein_sequence)

        # Translate each RNA sequence
        protein_sequences = {}
        for key, rna_sequence in rna_sequences.items():
            protein_sequences[key] = translate_rna_to_protein(rna_sequence)

        protein_0 = protein_sequences["rna_sequence0"]
        protein_1 = protein_sequences["rna_sequence1"]
        protein_2 = protein_sequences["rna_sequence2"]

        translated_record_0.seq = protein_0

        with open(f"{path}_0.fasta", "a") as output_handle_0:
            SeqIO.write(translated_record_0, output_handle_0, "fasta")

        translated_record_1.seq = protein_1

        with open(f"{path}_1.fasta", "a") as output_handle_1:
            SeqIO.write(translated_record_1, output_handle_1, "fasta")

        translated_record_2.seq = protein_2

        with open(f"{path}_2.fasta", "a") as output_handle_2:
            SeqIO.write(translated_record_2, output_handle_2, "fasta")


# Define the input fasta files and corresponding output paths
inputs = [
    # (
    #     "data/Drosophila_melanogaster/GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fna",
    #     "data1/Drosophila_melanogaster/output_amino",
    # ),
    # (
    #     "data/eastern_oyster/GCF_002022765.2_C_virginica-3.0_genomic.fna",
    #     "data1/eastern_oyster/output.amino",
    # ),
    ("data/Human/GCF_000001405.40_GRCh38.p14_genomic.fna", "data1/Human/output.amino"),
    # (
    #     "data/lion/GCF_018350215.1_P.leo_Ple1_pat1.1_genomic.fna",
    #     "data1/lion/output.amino",
    # ),
    # (
    #     "data/zebra_fish/GCF_000002035.6_GRCz11_genomic.fna",
    #     "data1/zebra_fish/output.amino",
    # ),
]


# Function to run the translation
def run_translation(input_fasta, output_path):
    translate_fasta_with_start(input_fasta, output_path)


if __name__ == "__main__":
    # Create a pool of worker processes
    with multiprocessing.Pool() as pool:
        # Map the inputs to the translation function using multiprocessing
        pool.starmap(run_translation, inputs)
