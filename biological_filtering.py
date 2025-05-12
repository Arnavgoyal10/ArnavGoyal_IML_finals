import pandas as pd
import os, sys

# --- Load the model’s positive predictions ---
data = pd.read_csv(sys.argv[2], header=None)

# 1. Cysteine Exclusion
def remove_cysteine(df):
    return df[~df[0].str.contains("C")]

# 2. Charge Constraint (sliding-window check for local cationic character)
def seq_check_pos_charge(seq):
    for i in range(len(seq) - 4):
        window = seq[i:i+5]
        if window.count("K") + window.count("R") > 3:
            return False
    return True

def filter_charge(df):
    return df[df[0].apply(seq_check_pos_charge)]

# 3. Hydrophobicity Constraint (avoid fully hydrophobic tripeptides)
HYDROPHOBIC = set(["F","V","I","W","L","A","M"])
def seq_check_hydrophobic(seq):
    for i in range(len(seq) - 2):
        tri = seq[i:i+3]
        if all(residue in HYDROPHOBIC for residue in tri):
            return False
    return True

def filter_hydrophobic(df):
    return df[df[0].apply(seq_check_hydrophobic)]

# 4. Repeating-Motif Filtering (no runs of three identical residues)
def seq_check_repeat(seq):
    for i in range(len(seq) - 2):
        if seq[i] == seq[i+1] == seq[i+2]:
            return False
    return True

def filter_repeats(df):
    return df[df[0].apply(seq_check_repeat)]

# --- Apply filters sequentially ---
data = remove_cysteine(data)
data = filter_charge(data)
data = filter_hydrophobic(data)
data = filter_repeats(data)

# --- Write out the filtered list ---
filtered_txt = f"./kat/altai_{sys.argv[3]}_biological_filter_kat"
data.to_csv(filtered_txt, header=False, index=False)

# --- Convert to FASTA for clustering ---
def convert_to_fasta(in_file, out_file):
    with open(in_file) as inp, open(out_file, "w") as outp:
        for idx, line in enumerate(inp, 1):
            seq = line.strip()
            outp.write(f">seq_{idx}\n{seq}\n")

fasta_in  = filtered_txt
fasta_out = f"kat/biological_filter_altai_kat_{sys.argv[3]}.fasta"
convert_to_fasta(fasta_in, fasta_out)

# 5. Redundancy Removal (CD-HIT clustering at 60% identity)
os.system(
    f"cd-hit -i {fasta_out} -c 0.6 -o kat/cdhit_altai_kat_{sys.argv[3]} -n 4"
)
