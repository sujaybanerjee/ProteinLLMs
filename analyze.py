import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class AnalyzePredictions:
    def __init__(self,output_dir = None):
        self.output_dir = output_dir
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
    
    def load_data(self, predicted_results_dir, cutoff):
        # Function to load predicted ATP-binding residues from model output files
        # Inputs:
        #       predicted_results_dir: folder containing per-protein prediction .tsv files
        #       cutoff: threshold for selecting predicted binding residues
        # Returns:
        #       a dataframe of residues that meet the cutoff
        bind_residues = []
        for pname in os.listdir(predicted_results_dir):
            protid = os.path.splitext(pname)[0]
            path = os.path.join(predicted_results_dir,pname)

            pred_prob_df = pd.read_csv(path, sep='\s+')  #whitespace issues
            pred_prob_df = pred_prob_df[pred_prob_df["Prob"] >=cutoff].copy()
            pred_prob_df["protid"] = protid
            bind_residues.append(pred_prob_df[["protid", "Index", "AA", "Prob"]])
        return pd.concat(bind_residues,ignore_index=True)

    def get_bindingSite_labels(self, protein_bindingSite_file):  
        # Function to load true ATP-binding site labels from a CSV file
        # Inputs:
        #       protein_bindingSite_file: CSV with protid and binding site index
        # Returns:
        #       a dataframe containing binding site indices 
        bind_site_df = pd.read_csv(protein_bindingSite_file, sep=',')
        bind_site_df = bind_site_df.rename(columns={"binding_site": "Index"})
        return bind_site_df[["protid", "Index"]]
    
    def calculate_metrics(self, predicted_results_dir, cutoff, binding_df):
        # Function to compute classification metrics at a given probability cutoff
        # Inputs:
        #       predicted_results_dir: folder containing prediction outputs
        #       cutoff: threshold for predicted positives
        #       binding_df: dataframe of true binding residues
        # Returns:
        #       accuracy, precision, true positive rate, false positive rate
        binding_df = binding_df.copy()
        binding_df["binding_site"] = binding_df["Index"].astype(int)
        binding_prots = set(binding_df["protid"])
        
        #all residues for these proteins
        all_df = self.load_data(predicted_results_dir,cutoff= 0.0)
        all_df = all_df[all_df["protid"].isin(binding_prots)]
        all_pairs = set(zip(all_df["protid"], all_df["Index"]))

        #true binding sites
        true_pairs = set(zip(binding_df["protid"], binding_df["Index"])) & all_pairs

        #binding site at cutoff
        pred_df = self.load_data(predicted_results_dir, cutoff)
        pred_df = pred_df[pred_df["protid"].isin(binding_prots)]
        pred_pairs = set(zip(pred_df["protid"], pred_df["Index"]))

        TP = len(true_pairs&pred_pairs)
        FP = len(pred_pairs-true_pairs)
        FN = len(true_pairs-pred_pairs)
        TN = len(all_pairs-true_pairs-pred_pairs)
        total = TP + TN + FP + FN
        accuracy = (TP + TN)/total 
        precision = TP/(TP + FP)
        tpr = TP/(TP + FN) 
        fpr = FP/(FP + TN) 
        return accuracy, precision, tpr, fpr

if __name__ == "__main__":
    predicted_results_dir = sys.argv[1]     
    binding_site_file = sys.argv[2]    
    output_dir = sys.argv[3]            

    analyzer = AnalyzePredictions(output_dir)

    # Load known labels
    binding_df = analyzer.get_bindingSite_labels(binding_site_file)

    # #histogram at cutoff=0.5
    cutoff = 0.9
    df = analyzer.load_data(predicted_results_dir, cutoff)
    df_p41240 = df[df["protid"] == "P41240"].copy()
    print("\nPredicted ATP-binding residues for P41240 at cutoff =", cutoff)
    print(df_p41240)
    # Extract indices as a list
    indices = df_p41240["Index"].astype(int).tolist()

    print("\nResidue indices for PyMOL selection:")
    if len(indices) > 0:
        # Format: "resi 10+15+20"
        pymol_sel = "resi " + "+".join(str(i) for i in indices)
        print(pymol_sel)
    else:
        print("No residues predicted at this cutoff.")

    counts = df["AA"].value_counts()  
    plt.bar(counts.index, counts.values)
    plt.xlabel("Amino Acid")
    plt.ylabel("Frequency Predicted as Binding Site")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "AA_histogram.png"))
    plt.close()

    #ROC curve over 10 cutoff points
    cutoffs = np.linspace(0, 1, 10)
    TPRs, FPRs = [], []

    for c in cutoffs:
        _, _, tpr, fpr = analyzer.calculate_metrics(predicted_results_dir, c, binding_df)
        TPRs.append(tpr)
        FPRs.append(fpr)

    plt.plot(FPRs, TPRs, marker="o")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ROC.png"))
    plt.close()

    print("Done! Saved AA_histogram.png and ROC.png")