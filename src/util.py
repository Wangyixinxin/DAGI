import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

def initialize_variables():
    """
    Initialize global lists and dictionaries.
    """
    regionaparc_name_list = ['bankssts', 'caudalanteriorcingulate', 'caudalmiddlefrontal', 'cuneus', 'entorhinal', 'fusiform', 'inferiorparietal', 'inferiortemporal', 'isthmuscingulate', 'lateraloccipital', 'lateralorbitofrontal', 'lingual', 'medialorbitofrontal', 'middletemporal', 'parahippocampal', 'paracentral', 'parsopercularis', 'parsorbitalis', 'parstriangularis', 'pericalcarine', 'postcentral', 'posteriorcingulate', 'precentral', 'precuneus', 'rostralanteriorcingulate', 'rostralmiddlefrontal', 'superiorfrontal', 'superiorparietal', 'superiortemporal', 'supramarginal', 'frontalpole', 'temporalpole', 'transversetemporal', 'insula']


    regionaseg_name_list = ['lateral_ventricle', 'inf_lat_vent', 'cerebellum_white_matter', 'cerebellum_cortex', 'thalamus_proper', 'caudate', 'putamen', 'pallidum', '3rd_ventricle', '4th_ventricle', 'brain_stem', 'hippocampus', 'amygdala', 'csf', 'wm_hypointensities', 'cc_posterior', 'cc_mid_posterior', 'cc_central', 'cc_mid_anterior', 'cc_anterior']


    region_name_dict = {0: 'bankssts', 1: 'caudalanteriorcingulate', 2: 'caudalmiddlefrontal', 3: 'cuneus', 4: 'entorhinal', 5: 'fusiform', 6: 'inferiorparietal', 7: 'inferiortemporal', 8: 'isthmuscingulate',
                        9: 'lateraloccipital', 10: 'lateralorbitofrontal', 11: 'lingual', 12: 'medialorbitofrontal', 13: 'middletemporal', 14: 'parahippocampal', 15: 'paracentral', 16: 'parsopercularis',
                        17: 'parsorbitalis', 18: 'parstriangularis', 19: 'pericalcarine', 20: 'postcentral', 21: 'posteriorcingulate', 22: 'precentral', 23: 'precuneus', 24: 'rostralanteriorcingulate',
                        25: 'rostralmiddlefrontal', 26: 'superiorfrontal', 27: 'superiorparietal', 28: 'superiortemporal', 29: 'supramarginal', 30: 'frontalpole', 31: 'temporalpole', 32: 'transversetemporal',
                        33: 'insula', 34: 'lateral_ventricle', 35: 'inf_lat_vent', 36: 'cerebellum_white_matter', 37: 'cerebellum_cortex', 38: 'thalamus_proper', 39: 'caudate', 40: 'putamen', 41: 'pallidum',
                        42: '3rd_ventricle', 43: '4th_ventricle', 44: 'brain_stem', 45: 'hippocampus', 46: 'amygdala', 47: 'csf', 48: 'wm_hypointensities', 49: 'cc_posterior', 50: 'cc_mid_posterior',
                        51: 'cc_central', 52: 'cc_mid_anterior', 53: 'cc_anterior'}


    return regionaparc_name_list, regionaseg_name_list, region_name_dict


def build_adjacency_matrices(regionaparc_name_list, regionaseg_name_list, aparc_only=True):
    """
    Create an adjacency matrix based on provided connections.
    """
    # Define the number of regions
    num_regions_aparc = 34
    num_regions_aseg = len(regionaseg_name_list)

    # Initialize the adjacency matrix as a square numpy array of zeros
    adjacency_matrix_aparc = np.zeros((num_regions_aparc, num_regions_aparc), dtype=int)
    connections_aparc = [(0, 6), (0, 13), (0, 28),
                  (1, 21),(1,24),(1,26),
                   (2,16),(2,22),(2,25),(2,26),
                  (3,9), (3,19), (3,23), (3,27),
                  (4,5),(4,14),(4,28),(4,31),
                  (5,4),(5,7),(5,9),(5,11),(5,14),(5,31),
                  (6,27),(6,9),(6,29),(6,13),(6,28),(6,0),
                  (7,5),(7,9),(7,13),
                  (8,11),(8,14),(8,21),(8,23),
                  (9,19),(9,11),(9,13),(9,27),
                  (10,12),(10,25),(10,26),(10,33),
                  (11,14),(11,19),(11,23),
                  (12,20),(12,26),(12,30),
                  (13,28),
                  (15,21),(15,23),(15,20),(15,22),(15,26),
                  (16,22),(16,25),(16,18),(16,33),
                  (17,25),(17,18),(17,33),(17,30),
                  (18,25),(18,33),
                  (19,23),
                  (20,22),(20,23),(20,27),(20,29),(20,33),
                  (21,23),(21,26),
                  (22,26),(22,33),
                  (23,27),
                  (24,26),
                  (25,26),(25,30),
                  (27,29),
                  (28,29),(28,31),(28,32),(28,33),
                  (29,32),(29,33),
                  (32,33)]

    connections_aseg = []
    #adjacency_matrix = np.random.randint(2, size=(num_regions, num_regions))
    #adjacency_matrix_aparc = np.ones((num_regions_aparc, num_regions_aparc), dtype=int)

    num_regions_aparc = len(regionaparc_name_list)
    num_regions_aseg = len(regionaseg_name_list)
    
    adjacency_matrix_aparc = np.zeros((num_regions_aparc, num_regions_aparc), dtype=int)
    for i, j in connections_aparc:
        adjacency_matrix_aparc[i, j] = 1
        adjacency_matrix_aparc[j, i] = 1

    adjacency_matrix_aseg = np.zeros((num_regions_aseg, num_regions_aseg), dtype=int)
    for i, j in connections_aseg:
        adjacency_matrix_aseg[i, j] = 1
        adjacency_matrix_aseg[j, i] = 1

    if aparc_only:
        adjacency_matrix = adjacency_matrix_aparc
        region_name_list = regionaparc_name_list
    else:
        adjacency_matrix = np.block([
            [adjacency_matrix_aparc, np.zeros((num_regions_aparc, num_regions_aseg))], 
            [np.zeros((num_regions_aseg, num_regions_aparc)), adjacency_matrix_aseg]
        ])
        region_name_list = regionaparc_name_list + regionaseg_name_list

    return adjacency_matrix, region_name_list



def load_features(file_path):
    """
    Load features from the CSV file.
    """
    ncanda_allfeat = pd.read_csv(file_path, delimiter=',', dtype=np.float32)
    G_label = ncanda_allfeat['sex'].to_numpy().reshape(-1, 1)
    return ncanda_allfeat, G_label


def create_graph(ncanda_allfeat, G_label, region_name_list, adjacency_matrix):
    """
    Convert the data into a graph format for PyTorch Geometric.
    """
    indices = np.where(adjacency_matrix == 1)
    G_adjacency_matrix = np.vstack(indices)

    G_list = []
    for index, row in ncanda_allfeat.iterrows():
        df = pd.DataFrame({'surfarea': [], 'greyvol': [], 'thickavg': [], 'meancurv': [], 'gauscurv': [], 'volume_mm3': []})
        for region in region_name_list:
            try:
                df.loc[len(df)] = [row[region + '_surfarea'], row[region + '_grayvol'], row[region + '_thickavg'], row[region + '_meancurv'], row[region + '_gauscurv'], 0]
            except:
                df.loc[len(df)] = [0, 0, 0, 0, 0, row[region + '_volume_mm3']]
        df.index = region_name_list
        df = df[['surfarea', 'greyvol','thickavg', 'meancurv']]
        node_features = df.to_numpy()
        G = Data(x=torch.from_numpy(node_features).type(torch.float32), edge_index=torch.from_numpy(G_adjacency_matrix), y=torch.from_numpy(G_label[index]).type(torch.LongTensor))
        G_list.append(G)

    return G_list


    
if __name__ == "__main__":
    # Initialization
    regionaparc_name_list, regionaseg_name_list, region_name_dict = initialize_variables()
    
    # Build Adjacency Matrices
    adjacency_matrix, region_name_list = build_adjacency_matrices(regionaparc_name_list, regionaseg_name_list)
    
    # Load Features
    file_path = '/Users/xxx.csv'
    ncanda_allfeat, G_label = load_features(file_path)
    G_list = create_graph(ncanda_allfeat, G_label, region_name_list, adjacency_matrix)
    print(G_list)
