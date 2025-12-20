import sys
import os
import numpy as np
import torch
from Bio import SeqIO
import esm
import pickle
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

class ESM_model:
    def __init__(self, fasta_file, output_dir, protein_family_file):
        self.fasta_file = fasta_file
        self.output_dir = output_dir
        self.protein_family_file = protein_family_file
        os.makedirs(self.output_dir, exist_ok=True)
        self.sequences = self.load_fasta(self.fasta_file)

    def load_fasta(self, fasta_file):
        #Function to load UniProt IDs and sequences from a FASTA file
        #Inputs:
        #       fasta_file: path to a FASTA file 
        #Returns:
        #       a list of (protein_id, sequence) tuples from  FASTA
        sequence_list = []
        with open(fasta_file) as fasta:
            for record in SeqIO.parse(fasta,"fasta"):
                header = record.id
                id_parts = header.split('|')
                uniprot_id = id_parts[1]
                sequence = str(record.seq)
                sequence_list.append((uniprot_id,sequence))
        return sequence_list

    def get_vectors(self, list_tuples_protId_seq):
        # Function to generate per-residue ESM2 embeddings for each protein
        # Inputs:
        #       list_tuples_protId_seq: list of (protein_id, sequence) tuples
        # Returns:
        #       a dictionary mapping protein_id to tensor of residue embeddings (1, seqlength, 1280)
        #loading ESM2
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()

        embeddings = {}
        #prep data
        for uniprot_id,sequence in list_tuples_protId_seq:
            data = [(uniprot_id,sequence)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]

            tokens_len = batch_lens[0]#.item()   
            per_residue = token_representations[0, 1:tokens_len - 1]  
            embeddings[uniprot_id] = per_residue.unsqueeze(0).cpu()

        return embeddings


    def pool_representations(dict_protId_embedding, mean_max_param):
        # Function to pool per-residue embeddings into one vector per protein
        # Inputs:
        #       dict_protId_embedding: dict of protein_id to per-residue embedding tensor
        #       mean_max_param: string selecting pooling method ('mean' or 'max')
        # Returns:
        #       a dictionary mapping protein_id to pooled numpy vector 
        pooled_embeddings = {}
        for uniprot_id, embedding in dict_protId_embedding.items():
            if mean_max_param =='mean':
                pooled_tensor = embedding.mean(dim=1) 
            elif mean_max_param =='max':
                pooled_tensor = embedding.max(dim=1).values
            pooled_vec = pooled_tensor.squeeze(0).numpy()
            pooled_embeddings[uniprot_id] = pooled_vec

        return pooled_embeddings
    
    def calculate_silhouette_score(family_data_file, pooled_vectors_dict):
        # Function to compute silhouette scores for each protein family
        # Inputs:
        #       family_data_file: CSV with protein IDs and family labels
        #       pooled_vectors_dict: dict of protein_id to pooled embedding vector
        # Returns:
        #       a dataframe with columns ['family','score'] for each family’s silhouette score
        fam_file = pd.read_csv(family_data_file)
        protid_col = fam_file.columns[0]
        family_col = fam_file.columns[1]
        proteins = list(pooled_vectors_dict.keys())

        X = np.vstack([pooled_vectors_dict[pid] for pid in proteins])
        families = fam_file[family_col].unique()
        rows = []
        for fam in families:
            # labels[i] = 1 if proteins[i] is in this family, 0 otherwise
            labels = []
            for protid in proteins:
                is_in_family = ((fam_file[protid_col] == protid) & (fam_file[family_col] == fam)).any()
                labels.append(1 if is_in_family else 0)
            labels = np.array(labels)
            score = silhouette_score(X, labels)
            rows.append({"family": fam, "score": score})
        score_df = pd.DataFrame(rows, columns=["family", "score"])
        return score_df
            

    def make_tsne_plot(pooled_vectors_dict, output_path, color_by_family=False, family_data_file=None):
        # Function to generate a 2D t-SNE plot of protein embeddings
        # Inputs:
        #       pooled_vectors_dict: dict of protein_id to pooled vector
        #       output_path: file path to save the plot
        #       color_by_family: whether to color points by protein family or not
        #       family_data_file: CSV file with protein_id to family labels
        # Returns:
        #       None, saves a t-SNE PNG image to output_path
        protein_ids = list(pooled_vectors_dict.keys())
        X = np.array(list(pooled_vectors_dict.values()))
        tsne = TSNE(n_components =2, random_state =42)
        coords = tsne.fit_transform(X)   

        #family-based coloring
        if color_by_family and family_data_file is not None:
            fam_file = pd.read_csv(family_data_file)
            protid_col = fam_file.columns[0]
            family_col = fam_file.columns[1]
            fam_map = dict(zip(fam_file[protid_col], fam_file[family_col]))
            labels = [fam_map.get(protid, "unknown fam") for protid in protein_ids] 
            codes, fam = pd.factorize(labels)
            plt.figure()
            plt.scatter(coords[:, 0], coords[:, 1], c=codes, s=10, cmap="GnBu")
        else:
            #reg tSNE
            plt.figure()
            plt.scatter(coords[:, 0], coords[:, 1], s=10)
        plt.xlabel("Component 1", fontsize=12)
        plt.ylabel("Component 2", fontsize=12)
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

        plt.xticks(np.linspace(x_min, x_max, 5))
        plt.yticks(np.linspace(y_min, y_max, 5))
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

if __name__ == "__main__":
    fasta_file = sys.argv[1]
    output_dir = sys.argv[2]       
    protein_family_file = sys.argv[3]

    model = ESM_model(fasta_file, output_dir, protein_family_file)
    print("Loaded",len(model.sequences), "sequences from",fasta_file)
    print("Output directory set:",model.output_dir)

    #per-token embeddings for embeddings.pkl
    print("Computing ESM embeddings:")
    embeddings = model.get_vectors(model.sequences)
    emb_path = os.path.join(model.output_dir,"embeddings.pkl")
    with open(emb_path,"wb") as file:
        pickle.dump(embeddings,file)
    print("Saved per-residue embeddings to:",emb_path)

    #pool representations w/ mean and max and pickle 
    print("Pooling embeddings for mean:")
    mean_pooled = ESM_model.pool_representations(embeddings,'mean')
    mean_pkl = os.path.join(model.output_dir,"meanPooled_embeddings.pkl")
    with open(mean_pkl, "wb") as f:
        pickle.dump(mean_pooled,f)
    print("Saved mean-pooled embeddings to:",mean_pkl)

    print("Pooling embeddings for max:")
    max_pooled = ESM_model.pool_representations(embeddings,'max')
    max_pkl = os.path.join(model.output_dir,"maxPooled_embeddings.pkl")
    with open(max_pkl, "wb") as f:
        pickle.dump(max_pooled,f)
    print("Saved max pooled embeddings to",max_pkl)

    #t-SNE plots reg.
    mean_tsne_path = os.path.join(model.output_dir, "meanPooled_viz.png")
    max_tsne_path = os.path.join(model.output_dir, "maxPooled_viz.png")
    print("Creating t-SNE plots (reg):")
    ESM_model.make_tsne_plot(mean_pooled, mean_tsne_path,color_by_family=False,family_data_file=None)
    ESM_model.make_tsne_plot(max_pooled, max_tsne_path,color_by_family=False, family_data_file=None)

    #Silhouette scores 
    print("Computing silhouette scores...")
    mean_sil_df = ESM_model.calculate_silhouette_score(protein_family_file,mean_pooled)
    max_sil_df = ESM_model.calculate_silhouette_score(protein_family_file,max_pooled)
    mean_sil_path = os.path.join(model.output_dir, "meanPooled_silhouette.csv")
    max_sil_path = os.path.join(model.output_dir, "maxPooled_silhouette.csv")
    mean_sil_df.to_csv(mean_sil_path, index= False)
    max_sil_df.to_csv(max_sil_path, index=False)
    print("Saved silhouette scores to", mean_sil_path, "and", max_sil_path)

    #t-SNE plots w/ family coloring
    mean_tsne_fam_path = os.path.join(model.output_dir, "meanPooled_viz_family.png")
    max_tsne_fam_path = os.path.join(model.output_dir, "maxPooled_viz_family.png")

    print("Creating t-SNE plots (colored by family):")
    ESM_model.make_tsne_plot(mean_pooled, mean_tsne_fam_path,color_by_family=True,family_data_file = protein_family_file)
    ESM_model.make_tsne_plot(max_pooled, max_tsne_fam_path,color_by_family=True, family_data_file= protein_family_file)

    print("Done!")


