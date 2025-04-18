import os
import pickle

data_dir = '/blob/dda/PXD028806/training_data/'
file_template = 'PXD028806_tailor_{}.pkl'
num_files = 11

# Final dictionary: {protein_id -> [sequences]}
protein_to_sequences = {}

for i in range(1, num_files + 1):
    file_path = os.path.join(data_dir, file_template.format(i))
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

        if hasattr(data, 'to_dict'):
            data = data.to_dict(orient='records')

        for entry in data:
            if not isinstance(entry, dict):
                print('Error')  # skip unexpected format

            protein_ids = entry.get('protein_ids', {})
            sequence = entry.get('sequence', None)

            if sequence and isinstance(protein_ids, dict):
                for protein, positions in protein_ids.items():
                    if protein not in protein_to_sequences:
                        protein_to_sequences[protein] = []
                    protein_to_sequences[protein].append(sequence)

with open('protein_seq_dict.pkl', 'wb') as out_file:
     pickle.dump(protein_to_sequences, out_file)

print(f"Total unique proteins: {len(protein_to_sequences)}")

for prot, seqs in list(protein_to_sequences.items())[:5]:
    print(f"{prot}: {len(seqs)} sequences")