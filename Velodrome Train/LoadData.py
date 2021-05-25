import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

def mapping(data_file, mapping_table, exclude_many_to_one = True):
    '''data_file - path to data matrix file, with Entrez ID in the first column.
    mapping_table - pandas dataframe with "ensg_v99" and "entrez_v99" columns -- Ensembl and Entrez gene ids for Ensembl v99.
    exclude_many_to_one - whether to exclude genes which Entrez ids mapped to a non-unique Ensemble. 
    If False, for each Ensembl id, sum of rows of with correponding Entrez ids is reported (makes sense for expression data)'''
    # reading inputs
    data = pd.read_csv(data_file,sep="\t",index_col=0) 
#     print("Gene ids in data matrix:",data.shape[0])

    # genes not found in mapping table
    not_mapped = set(data.index.values).difference(set(mapping_table["entrez_v99"].values))
#     print("\tnot found in mapping table:",len(not_mapped))
    mapping_table = mapping_table.loc[mapping_table["entrez_v99"].isin(set(mapping_table["entrez_v99"].values).intersection(set(data.index.values))),:]
    data = data.loc[sorted(set(data.index.values).difference(not_mapped)),:]
#     print("Found in mapping table:",data.shape[0])
    
    # identifying genes mapped as one-to-many:  
    entrez_one_to_many = mapping_table.groupby("entrez_v99")[["ensg_v99"]].agg('count')
    entrez_one_to_many = set(entrez_one_to_many.loc[entrez_one_to_many["ensg_v99"]>1,:].sort_values("ensg_v99").index.values)
#     print("one-to-many (will be dropped):")
#     print("\t%s Entrez ids -> %s Ensemble ids"%(len(entrez_one_to_many),
#                                                 mapping_table.loc[mapping_table["entrez_v99"].isin(entrez_one_to_many),:].shape[0]))

    # identifying genes mapped as many-to-one:  
    many_to_one = mapping_table.loc[~mapping_table["entrez_v99"].isin(entrez_one_to_many)].groupby("ensg_v99")[["entrez_v99"]].agg('count')
    many_to_one = set(many_to_one.loc[many_to_one["entrez_v99"]>1,:].sort_values("entrez_v99").index.values)
#     if exclude_many_to_one:
#         print("many-to-one (will be dropped):")
#     else:
#         print("many-to-one (will be summed and mapped):")
    entrez_many_to_one = set(mapping_table.loc[mapping_table["ensg_v99"].isin(many_to_one),"entrez_v99"].values)
#     print("\t%s Entrez ids -> %s Ensemble ids"%(len(entrez_many_to_one),len(many_to_one)))
    many_to_one_mapping = mapping_table.loc[mapping_table["ensg_v99"].isin(many_to_one),["entrez_v99","ensg_v99"]]

    
    # one-to-one mapping
    #print("one-to-one (will be mapped):")
    one_to_one_mapping = mapping_table.loc[~mapping_table["entrez_v99"].isin(entrez_one_to_many)]
    one_to_one_mapping = one_to_one_mapping.loc[~one_to_one_mapping["ensg_v99"].isin(many_to_one)]
#     print("\t",one_to_one_mapping.shape[0])
    one_to_one_mapping = one_to_one_mapping[["entrez_v99","ensg_v99"]]
    one_to_one_mapping = one_to_one_mapping.set_index("entrez_v99").to_dict()['ensg_v99']
    d1 = data.loc[sorted(one_to_one_mapping.keys()),:]
    d1.rename(one_to_one_mapping,axis="index",inplace=True)
    
    # many-to-one mapping 
    if not exclude_many_to_one:
        d2=data.loc[sorted(entrez_many_to_one),:]
        many_to_one_mapping = many_to_one_mapping.groupby("ensg_v99")[["entrez_v99"]].agg(list).to_dict()['entrez_v99']
        d2_mapped = {}
        for ensg in many_to_one_mapping.keys():
            entrez_genes = many_to_one_mapping[ensg]
            d2_mapped[ensg] = d2.loc[entrez_genes,:].sum()
        d2 = pd.DataFrame.from_dict(d2_mapped).T

        data = pd.concat([d1,d2],axis=0)
    else:
        data = d1
    return data


def prep_data(args):
    
    drug = args.drug
        
    CTRP_exprs = pd.read_csv(args.data_root + "CTRP.exprsALL.tsv", sep = "\t", index_col=0)
    GDSC_exprs = pd.read_csv(args.data_root + "GDSCv2.exprsALL.tsv", sep = "\t", index_col=0)
    gCSI_exprs = pd.read_csv(args.data_root + "gCSI.exprsALL.tsv", sep = "\t", index_col=0)

    CTRP_aac = pd.read_csv(args.data_root + "CTRP.aacALL.tsv", sep = "\t", index_col=0)
    GDSC_aac = pd.read_csv(args.data_root + "GDSCv2.aacALL.tsv", sep = "\t", index_col=0)
    gCSI_aac = pd.read_csv(args.data_root + "gCSI.aacALL.tsv", sep = "\t", index_col=0)

    CTRP_info = pd.read_csv(args.data_root + "CTRP.infoALL.tsv", sep = "\t", index_col=0)
    idx_other_ctrp = CTRP_info.index[CTRP_info["Tumor"] == 1]
    GDSC_info = pd.read_csv(args.data_root + "GDSCv2.infoALL.tsv", sep = "\t", index_col=0)
    idx_other_gdsc = GDSC_info.index[GDSC_info["Tumor"] == 1]
    gCSI_info = pd.read_csv(args.data_root + "gCSI.infoALL.tsv", sep = "\t", index_col=0)
    idx_other_gcsi = gCSI_info.index[gCSI_info["Tumor"] == 1]
    
    genes_1 = pd.read_csv(args.data_root + "gene_list_4Feb.mapped_to_ens99.tsv", sep = "\t", index_col=None)
    genes_2 = pd.read_csv(args.data_root + "gene_list_4Feb.mapped_to_ens99_part2.tsv", sep = "\t", index_col=None)
    list_genes = pd.concat([genes_1["ensg_v99"], genes_2["ensg_v99"]]).values

    list_ctrp = CTRP_exprs.index.intersection(list_genes)
    CTRP_exprs = CTRP_exprs.loc[list_ctrp]

    list_gdsc = GDSC_exprs.index.intersection(list_genes)
    GDSC_exprs = GDSC_exprs.loc[list_gdsc]

    list_gcsi = gCSI_exprs.index.intersection(list_genes)
    gCSI_exprs = gCSI_exprs.loc[list_gcsi]
    
    mapping_table = pd.read_csv(args.data_root + "Entrez_to_Ensg99.mapping_table.tsv",sep="\t",index_col=0)        
    
    TCGA_LUAD = mapping(args.data_root + "TCGA-LUAD_exprs.z.tsv", mapping_table)
    TCGA_LUAD = TCGA_LUAD.dropna()
    TCGA_BRCA = mapping(args.data_root + "TCGA-BRCA_exprs.z.tsv", mapping_table)
    TCGA_BRCA = TCGA_BRCA.dropna()
    TCGA_PAAD = mapping(args.data_root + "TCGA-PAAD_exprs.z.tsv", mapping_table)
    TCGA_PAAD = TCGA_PAAD.dropna()
    
    idx = CTRP_exprs.index
    idx_LUAD = TCGA_LUAD.index
    idx = idx.intersection(idx_LUAD)
    idx_BRCA = TCGA_BRCA.index
    idx = idx.intersection(idx_BRCA)    
    idx_PAAD = TCGA_PAAD.index
    idx = idx.intersection(idx_PAAD)
      
    if drug == "Erlotinib":
        
        X_tr = []
        Y_tr = []
                
        GSE33072 = mapping(args.data_root + "GSE33072_exprs.Erlotinib.tsv", mapping_table)
        idx_GSE33072 = GSE33072.index
        PDX_Erlotinib = mapping(args.data_root + "PDX_exprs.Erlotinib.tsv", mapping_table)        
        idx_PDXErlo = PDX_Erlotinib.index
        
        GSE33072R = pd.read_csv("Data/GSE33072_response.Erlotinib.tsv", sep = "\t", index_col=0, decimal = ",")
        PDX_ErloR = pd.read_csv("Data/PDX_response.multi-OMICS.Erlotinib.tsv", sep = "\t", index_col=0, decimal = ",")
        
        idx = idx.intersection(idx_GSE33072)
        idx = idx.intersection(idx_PDXErlo)
        
        ls1 = GSE33072.columns.intersection(GSE33072R.index)
        ls2 = PDX_Erlotinib.columns.intersection(PDX_ErloR.index)        
        
        GSE33072 = pd.DataFrame.transpose(GSE33072.loc[idx,ls1])
        PDX_Erlotinib = pd.DataFrame.transpose(PDX_Erlotinib.loc[idx,ls2])
        
        GSE33072R = GSE33072R.loc[ls1,:]
        PDX_ErloR = PDX_ErloR.loc[ls2,:]
        GSE33072R.loc[GSE33072R.iloc[:,1] == 'R'] = 0
        GSE33072R.loc[GSE33072R.iloc[:,1] == 'S'] = 1
        GSE33072R = GSE33072R["response"].values
        PDX_ErloR.loc[PDX_ErloR.iloc[:,1] == 'R'] = 0
        PDX_ErloR.loc[PDX_ErloR.iloc[:,1] == 'S'] = 1
        PDX_ErloR = PDX_ErloR["response"].values     
        
        CTRP_exprs = CTRP_exprs.loc[idx,:] 
        GDSC_exprs = GDSC_exprs.loc[idx,:]
        gCSI_exprs = gCSI_exprs.loc[idx,:]
        TCGA_LUAD = pd.DataFrame.transpose(TCGA_LUAD.loc[idx,:])
        TCGA_BRCA = pd.DataFrame.transpose(TCGA_BRCA.loc[idx,:])
        TCGA_PAAD = pd.DataFrame.transpose(TCGA_PAAD.loc[idx,:])        
        
        CTRP_aac_drug = CTRP_aac.loc[drug].dropna()
        GDSC_aac_drug = GDSC_aac.loc[drug].dropna()
        gCSI_aac_drug = gCSI_aac.loc[drug].dropna()

        idx_ctrp = CTRP_exprs.columns.intersection(CTRP_aac_drug.index)
        idx_ctrp = [x for x in idx_ctrp if x not in idx_other_ctrp]    
        idx_gdsc = GDSC_exprs.columns.intersection(GDSC_aac_drug.index)
        idx_gdsc = [x for x in idx_gdsc if x not in idx_other_gdsc]    
        idx_gcsi = gCSI_exprs.columns.intersection(gCSI_aac_drug.index)
        idx_gcsi = [x for x in idx_gcsi if x not in idx_other_gcsi] 

        CTRP_exprs_drug = pd.DataFrame.transpose(CTRP_exprs.loc[:,idx_ctrp])
        CTRP_aac_drug = CTRP_aac_drug.loc[idx_ctrp]
        GDSC_exprs_drug = pd.DataFrame.transpose(GDSC_exprs.loc[:,idx_gdsc])
        GDSC_aac_drug = GDSC_aac_drug.loc[idx_gdsc]
        gCSI_exprs_drug = pd.DataFrame.transpose(gCSI_exprs.loc[:,idx_gcsi])
        gCSI_aac_drug = gCSI_aac_drug.loc[idx_gcsi]        
        
        X_tr.append(CTRP_exprs_drug.values)
        X_tr.append(GDSC_exprs_drug.values)
        
        Y_tr.append(CTRP_aac_drug.values)
        Y_tr.append(GDSC_aac_drug.values)        
        
        X_U = pd.concat([TCGA_LUAD, TCGA_BRCA, TCGA_PAAD], axis = 0).values

        return X_tr, Y_tr, gCSI_exprs_drug.values, gCSI_aac_drug.values, GSE33072.values, GSE33072R, PDX_Erlotinib.values, PDX_ErloR, X_U 
        
    if drug == "Docetaxel":  
        
        X_tr = []
        Y_tr = []        
        
        GSE25065D = mapping(args.data_root + "GSE25065_exprs.Docetaxel.tsv", mapping_table)
        idx_GSE25065D = GSE25065D.index

        GSE25065DR = pd.read_csv(args.data_root + "GSE25065_response.Docetaxel.tsv", sep = "\t", index_col=0, decimal = ",")        
        
        idx = idx.intersection(idx_GSE25065D)
        
        ls1 = GSE25065D.columns.intersection(GSE25065DR.index)
        
        GSE25065D = pd.DataFrame.transpose(GSE25065D.loc[idx,ls1])        
        
        GSE25065DR = GSE25065DR.loc[ls1,:]
        GSE25065DR.loc[GSE25065DR.iloc[:,1] == 'R'] = 0
        GSE25065DR.loc[GSE25065DR.iloc[:,1] == 'S'] = 1
        GSE25065DR = GSE25065DR["response"].values
        
        CTRP_exprs = CTRP_exprs.loc[idx,:] 
        GDSC_exprs = GDSC_exprs.loc[idx,:]
        gCSI_exprs = gCSI_exprs.loc[idx,:]
        TCGA_LUAD = pd.DataFrame.transpose(TCGA_LUAD.loc[idx,:])
        TCGA_BRCA = pd.DataFrame.transpose(TCGA_BRCA.loc[idx,:])
        TCGA_PAAD = pd.DataFrame.transpose(TCGA_PAAD.loc[idx,:])        
        
        CTRP_aac_drug = CTRP_aac.loc[drug].dropna()
        GDSC_aac_drug = GDSC_aac.loc[drug].dropna()
        gCSI_aac_drug = gCSI_aac.loc[drug].dropna()

        idx_ctrp = CTRP_exprs.columns.intersection(CTRP_aac_drug.index)
        idx_ctrp = [x for x in idx_ctrp if x not in idx_other_ctrp]    
        idx_gdsc = GDSC_exprs.columns.intersection(GDSC_aac_drug.index)
        idx_gdsc = [x for x in idx_gdsc if x not in idx_other_gdsc]    
        idx_gcsi = gCSI_exprs.columns.intersection(gCSI_aac_drug.index)
        idx_gcsi = [x for x in idx_gcsi if x not in idx_other_gcsi] 

        CTRP_exprs_drug = pd.DataFrame.transpose(CTRP_exprs.loc[:,idx_ctrp])
        CTRP_aac_drug = CTRP_aac_drug.loc[idx_ctrp]
        GDSC_exprs_drug = pd.DataFrame.transpose(GDSC_exprs.loc[:,idx_gdsc])
        GDSC_aac_drug = GDSC_aac_drug.loc[idx_gdsc]
        gCSI_exprs_drug = pd.DataFrame.transpose(gCSI_exprs.loc[:,idx_gcsi])
        gCSI_aac_drug = gCSI_aac_drug.loc[idx_gcsi]     
        
        X_tr.append(CTRP_exprs_drug.values)
        X_tr.append(GDSC_exprs_drug.values)
        
        Y_tr.append(CTRP_aac_drug.values)
        Y_tr.append(GDSC_aac_drug.values)        

        X_U = pd.concat([TCGA_LUAD, TCGA_BRCA, TCGA_PAAD], axis = 0).values
                        
        return X_tr, Y_tr, gCSI_exprs_drug.values, gCSI_aac_drug.values, GSE25065D.values, GSE25065DR, X_U
        
    if drug == "Gemcitabine":    
        
        X_tr = []
        Y_tr = []        
        
        TCGA_Gemcitabine = mapping(args.data_root + "TCGA_exprs.Gemcitabine.tsv", mapping_table)
        idx_TCGA_Gemcitabine = TCGA_Gemcitabine.index
        PDX_Gemcitabine = mapping(args.data_root + "PDX_exprs.Gemcitabine.tsv", mapping_table)
        idx_PDX_Gemcitabine = PDX_Gemcitabine.index
        
        TCGA_GemR = pd.read_csv(args.data_root + "TCGA_response.Gemcitabine.tsv", sep = "\t", index_col=0, decimal = ",")
        PDX_GemR = pd.read_csv(args.data_root + "PDX_response.multi-OMICS.Gemcitabine.tsv", sep = "\t", index_col=0, decimal = ",")
                
        idx = idx.intersection(idx_TCGA_Gemcitabine)
        idx = idx.intersection(idx_PDX_Gemcitabine)
        
        ls1 = TCGA_Gemcitabine.columns.intersection(TCGA_GemR.index)
        ls2 = PDX_Gemcitabine.columns.intersection(PDX_GemR.index)            
        
        TCGA_Gemcitabine = pd.DataFrame.transpose(TCGA_Gemcitabine.loc[idx,ls1])
        PDX_Gemcitabine = pd.DataFrame.transpose(PDX_Gemcitabine.loc[idx,ls2])    
        
        TCGA_GemR = TCGA_GemR.loc[ls1,:]
        PDX_GemR = PDX_GemR.loc[ls2,:]
        TCGA_GemR.loc[TCGA_GemR.iloc[:,1] == 'R'] = 0
        TCGA_GemR.loc[TCGA_GemR.iloc[:,1] == 'S'] = 1
        TCGA_GemR = TCGA_GemR["response"].values
        PDX_GemR.loc[PDX_GemR.iloc[:,1] == 'R'] = 0
        PDX_GemR.loc[PDX_GemR.iloc[:,1] == 'S'] = 1
        PDX_GemR = PDX_GemR["response"].values  
        
        CTRP_exprs = CTRP_exprs.loc[idx,:] 
        GDSC_exprs = GDSC_exprs.loc[idx,:]
        gCSI_exprs = gCSI_exprs.loc[idx,:]
        TCGA_LUAD = pd.DataFrame.transpose(TCGA_LUAD.loc[idx,:])
        TCGA_BRCA = pd.DataFrame.transpose(TCGA_BRCA.loc[idx,:])
        TCGA_PAAD = pd.DataFrame.transpose(TCGA_PAAD.loc[idx,:])        
        
        CTRP_aac_drug = CTRP_aac.loc[drug].dropna()
        GDSC_aac_drug = GDSC_aac.loc[drug].dropna()
        gCSI_aac_drug = gCSI_aac.loc[drug].dropna()

        idx_ctrp = CTRP_exprs.columns.intersection(CTRP_aac_drug.index)
        idx_ctrp = [x for x in idx_ctrp if x not in idx_other_ctrp]    
        idx_gdsc = GDSC_exprs.columns.intersection(GDSC_aac_drug.index)
        idx_gdsc = [x for x in idx_gdsc if x not in idx_other_gdsc]    
        idx_gcsi = gCSI_exprs.columns.intersection(gCSI_aac_drug.index)
        idx_gcsi = [x for x in idx_gcsi if x not in idx_other_gcsi] 

        CTRP_exprs_drug = pd.DataFrame.transpose(CTRP_exprs.loc[:,idx_ctrp])
        CTRP_aac_drug = CTRP_aac_drug.loc[idx_ctrp]
        GDSC_exprs_drug = pd.DataFrame.transpose(GDSC_exprs.loc[:,idx_gdsc])
        GDSC_aac_drug = GDSC_aac_drug.loc[idx_gdsc]
        gCSI_exprs_drug = pd.DataFrame.transpose(gCSI_exprs.loc[:,idx_gcsi])
        gCSI_aac_drug = gCSI_aac_drug.loc[idx_gcsi]         
        
        X_tr.append(CTRP_exprs_drug.values)
        X_tr.append(GDSC_exprs_drug.values)
        
        Y_tr.append(CTRP_aac_drug.values)
        Y_tr.append(GDSC_aac_drug.values)
        
        X_U = pd.concat([TCGA_LUAD, TCGA_BRCA, TCGA_PAAD], axis = 0).values

        return X_tr, Y_tr, gCSI_exprs_drug.values, gCSI_aac_drug.values, TCGA_Gemcitabine.values, TCGA_GemR, PDX_Gemcitabine.values, PDX_GemR, X_U                     
        
    if drug == "Paclitaxel":     
        
        X_tr = []
        Y_tr = []        
        
        PDX_Paclitaxel = mapping(args.data_root + "PDX_exprs.Paclitaxel.tsv", mapping_table)
        idx_PDX_Paclitaxel = PDX_Paclitaxel.index
        GSE25065P = mapping(args.data_root + "GSE25065_exprs.Paclitaxel.tsv", mapping_table)
        idx_GSE25065P = GSE25065P.index
        
        PDX_PacR = pd.read_csv(args.data_root + "PDX_response.multi-OMICS.Paclitaxel.tsv", sep = "\t", index_col=0, decimal = ",")
        GSE25065PR = pd.read_csv(args.data_root + "GSE25065_response.Paclitaxel.tsv", sep = "\t", index_col=0, decimal = ",")
        
        idx = idx.intersection(idx_PDX_Paclitaxel)
        idx = idx.intersection(idx_GSE25065P)
        
        ls1 = PDX_Paclitaxel.columns.intersection(PDX_PacR.index)
        ls2 = GSE25065P.columns.intersection(GSE25065PR.index)         
        
        PDX_Paclitaxel = pd.DataFrame.transpose(PDX_Paclitaxel.loc[idx,ls1])
        GSE25065P = pd.DataFrame.transpose(GSE25065P.loc[idx,ls2])          

        PDX_PacR = PDX_PacR.loc[ls1,:]
        GSE25065PR = GSE25065PR.loc[ls2,:]
        PDX_PacR.loc[PDX_PacR.iloc[:,1] == 'R'] = 0
        PDX_PacR.loc[PDX_PacR.iloc[:,1] == 'S'] = 1
        PDX_PacR = PDX_PacR["response"].values
        GSE25065PR.loc[GSE25065PR.iloc[:,1] == 'R'] = 0
        GSE25065PR.loc[GSE25065PR.iloc[:,1] == 'S'] = 1
        GSE25065PR = GSE25065PR["response"].values
        
        CTRP_exprs = CTRP_exprs.loc[idx,:] 
        GDSC_exprs = GDSC_exprs.loc[idx,:]
        gCSI_exprs = gCSI_exprs.loc[idx,:]
        TCGA_LUAD = pd.DataFrame.transpose(TCGA_LUAD.loc[idx,:])
        TCGA_BRCA = pd.DataFrame.transpose(TCGA_BRCA.loc[idx,:])
        TCGA_PAAD = pd.DataFrame.transpose(TCGA_PAAD.loc[idx,:])        
        
        CTRP_aac_drug = CTRP_aac.loc[drug].dropna()
        GDSC_aac_drug = GDSC_aac.loc[drug].dropna()
        gCSI_aac_drug = gCSI_aac.loc[drug].dropna()

        idx_ctrp = CTRP_exprs.columns.intersection(CTRP_aac_drug.index)
        idx_ctrp = [x for x in idx_ctrp if x not in idx_other_ctrp]    
        idx_gdsc = GDSC_exprs.columns.intersection(GDSC_aac_drug.index)
        idx_gdsc = [x for x in idx_gdsc if x not in idx_other_gdsc]    
        idx_gcsi = gCSI_exprs.columns.intersection(gCSI_aac_drug.index)
        idx_gcsi = [x for x in idx_gcsi if x not in idx_other_gcsi] 

        CTRP_exprs_drug = pd.DataFrame.transpose(CTRP_exprs.loc[:,idx_ctrp])
        CTRP_aac_drug = CTRP_aac_drug.loc[idx_ctrp]
        GDSC_exprs_drug = pd.DataFrame.transpose(GDSC_exprs.loc[:,idx_gdsc])
        GDSC_aac_drug = GDSC_aac_drug.loc[idx_gdsc]
        gCSI_exprs_drug = pd.DataFrame.transpose(gCSI_exprs.loc[:,idx_gcsi])
        gCSI_aac_drug = gCSI_aac_drug.loc[idx_gcsi]      
        
        X_tr.append(CTRP_exprs_drug.values)
        X_tr.append(GDSC_exprs_drug.values)
        
        Y_tr.append(CTRP_aac_drug.values)
        Y_tr.append(GDSC_aac_drug.values)
        
        X_U = pd.concat([TCGA_LUAD, TCGA_BRCA, TCGA_PAAD], axis = 0).values

        return X_tr, Y_tr, gCSI_exprs_drug.values, gCSI_aac_drug.values, GSE25065P.values, GSE25065PR, PDX_Paclitaxel.values, PDX_PacR, X_U 
