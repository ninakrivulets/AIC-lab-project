import pickle
from Bio import SeqIO

# defining class ProteinObj
class ProteinObj:
    def __init__(self, index, protein_id, sequence):
        self.index = index
        self.protein_id = protein_id
        self.sequence = sequence

    def __repr__(self):
        return f"ProteinObj({self.index}, {self.protein_id}, {self.sequence[:10]}...)"  
    
# Class for working with a collection
class ProteinCollection:
    def __init__(self):
        self.protein_collection = []
        self.long_sequence = ""  # String for merged sequence
        self.position_dict = {}  # Dictionary for start positions

    def load_fasta(self, path_to_fasta):
        cnt = 0
        for record in SeqIO.parse(path_to_fasta, "fasta"):
            # add protein
            self.protein_collection.append(
                ProteinObj(cnt, str(record.id), str(record.seq))
            )
            # adding protein to the sequence
            self.long_sequence += str(record.seq)
            # saving the starting position
            self.position_dict[str(record.id)] = len(self.long_sequence) - len(str(record.seq))
            cnt += 1
            print('Finished with', cnt)

    def save_to_pickle(self, file_path):
        # saving the merged sequence and the dict to pickle file
        with open(file_path, 'wb') as f:
            pickle.dump({'long_sequence': self.long_sequence, 'position_dict': self.position_dict}, f)

    def __repr__(self):
        return f"ProteinCollection with {len(self.protein_collection)} proteins"

protein_collector = ProteinCollection()

path_to_fasta = "/home/data/Fasta/uniprot-proteome_UP000005640_canonical_isoforms.fasta"

# loading data from fasta file
protein_collector.load_fasta(path_to_fasta)

pickle_file_path = "/home/ninak/data/protein_data.pkl"
protein_collector.save_to_pickle(pickle_file_path)


print(f"Downloaded {len(protein_collector.protein_collection)} proteins.")
print(f"Length of the sequence: {len(protein_collector.long_sequence)}")
print("Positions of starts of the first five proteins:")
for protein in list(protein_collector.position_dict.items())[:5]:
    print(protein)
