import pandas as pd
import numpy as np
import math
from functools import reduce
import pd_extension
from datetime import datetime

"""
This module is to process annotation files.
First call merge_annotations() to merge different annotation sources together to
get a pd.DataFrame. 

"""


#################################################################################
## read in annotation files from different sources, and merge them into dataframe
## and then process and clean up the dataframe. 
## merge_annotations()
#################################################################################

def read_general(file, header=0, compression='infer', fixChr = False, indexRange = range(4), renameDict = None):
    df = pd.read_csv(file, delimiter='\t', header=header, compression = compression, na_values='.', low_memory = False)
    if fixChr:
        df.iloc[:,0]= df.apply(lambda x: x[0][3:], axis = 1)
    new_idx = df.apply(lambda x: '-'.join([str(x[i]) for i in indexRange]), axis = 1)
    df.set_index(new_idx, inplace = True)
    if renameDict is not None:
        df.rename(columns = renameDict, inplace = True)
    return df.iloc[:,(indexRange[-1]+1):]

def read_annovar(file):
    return read_general(file, indexRange = [0,1,3,4])

def read_deepsea(file):
    df = read_general(file, header=None, fixChr = True, renameDict = {4: 'deepsea'})
    df['deepsea_mlog2'] = df.apply(lambda x: -math.log2(x['deepsea']), axis = 1)
    return df

def read_wgsa(file):
    return read_general(file, compression='gzip')

def add_position(df, idx = None):
    if idx is None:
        idx = list(df.index)
    d = pd.DataFrame(list(map(lambda x: x.split('-'), idx)), columns=['chr','pos','ref','alt'], index=idx)
    return pd.merge(d, df, left_index = True, right_index = True, how = 'outer')

def merge_annotations(annotationDict, keepPosition = True, add_variant_type = True, clean_chr = True, process = True):
    """

    annotationDict: provide a dictionary of {dataname: filename}   
    	recognized dataname is: wgsa, deepsea, annovar. Otherwise will be
        in general format.
    
    Parameters:
    -----------
    annotationDict   : a dict contains {dataname : file_locaiton}
    keepPosition     : whether to add chr,pos,ref,alt to the returning dataframe
    add_variant_type : whether add variant type (SNV, deletion, insertion, MNT_change)
    clean_chr        : whether only keep chr1-22,X,Y. If you want to customize this,
                       you need to run pd.DataFrame.clean_chr() afterwards

    """
    knownDataname = ['wgsa','deepsea','annovar']
    dfs = []
    for dataname,file in annotationDict.items():
        if dataname not in knownDataname:
            func = read_general
        else:
            func = globals()['read_' + dataname]
        dfs.append(func(file))

    df = reduce(lambda left,right: pd.merge(left, right, left_index = True,right_index = True,how = 'outer'), dfs)

    if keepPosition or add_variant_type or clean_chr:
        df = add_position(df)
    if add_variant_type:
        df.add_variant_type(inplace = True)
    if clean_chr:
        df.clean_chr(inplace = True)
    if not keepPosition:
        df.drop(['chr','pos','ref','alt'],axis = 1, inplace = True)
    if process:
        process_annotation(df)

    return df


#############################################################################
## recode annotation columns that contains mulitiple records separated by '|'
#############################################################################

# These functions will be used in recode_annotation(df)
# for removing duplicate records such as from ANNOVAR_ensembl_Effect
def remove_dup(s, sep = '|'):
    if s == s:
        l = s.split(sep)
        return sorted(list(set(l)))
    else:
        return np.nan

# for removing duplicate records such as molecular_consequence
def remove_dup_mc(s, first = ',', second = '|', keepIndex = 1):
    if s == s:
        fir = s.split(first)
        sec = [x.split(second)[keepIndex] for x in fir]
        return sorted(list(set(sec)))
    else:
        return np.nan
    
# days from last evaluation to 2018-02-01
def days_till(d, till = 'Feb 01, 2018'):
    if (d == d) and (d != '-'):
        now = datetime.strptime(till, '%b %d, %Y')
        then = datetime.strptime(d, '%b %d, %Y')
        return (now - then).days
    else:
        return np.nan
    
def string_of_recoded_variable(df, column):
    return df[column].apply(lambda x: '-'.join(x) if x == x else 'NA')

def recode_annotation(df):
    """Some columns have input separated by '|', and have duplicates.
    This function is to recode these columns into ordered list,
    without duplicate elements"""
    
    # recode the effect of variant by ensembl (ANNOVAR_ensembl_Effect)
    if 'ANNOVAR_ensembl_Effect' in df.columns:
        df['effect'] = df.apply(lambda x: remove_dup(x['ANNOVAR_ensembl_Effect']), axis = 1)
        df['effect_str'] = string_of_recoded_variable(df, 'effect')

    # recode molecular_consequence
    if 'molecular_consequence' in df.columns:
        df['molecular_consequence_recode'] = df.apply(lambda x: remove_dup_mc(x['molecular_consequence']), axis = 1)
        df['mc_str'] = string_of_recoded_variable(df, 'molecular_consequence_recode')

    # recode LastEvaluated to days till 2018-02-01
    if 'LastEvaluated' in df.columns:
        df['days'] = df.apply(lambda x: days_till(x['LastEvaluated']), axis = 1)
    

#######################################################################################
## get the type of variant (coding, non_coding, or splicing) from effect and mc columns
#######################################################################################

def determine_coding_from_legendDict(x, legendDict):
    """
    This funtion is to determine whether the variant is coding / non-coding / splicing
    based on the annotation from ensemble or molecular consequence.
    LegendDict can be either effect_to_coding_legend or mc_to_coding_legend.
    
    This function is to apply to each row.
    """
    
    if legendDict == 'effect_to_coding_legend':
        legendDict = {'UTR3': 0,
         'UTR5': 0,
         'downstream': 0,
         'exonic': 1,
         'intergenic': 0,
         'intronic': 0,
         'ncRNA_exonic': 3,
         'ncRNA_intronic': 3,
         'ncRNA_splicing': 3,
         'nonsynonymous': 1,
         'splicing': 2,
         'stopgain': 1,
         'stoploss': 1,
         'synonymous': 1,
         'upstream': 0,
	 'nonframeshift': 0}
    elif legendDict == 'mc_to_coding_legend':
        legendDict = {'2KB_upstream_variant': 0,
         '3_prime_UTR_variant': 0,
         '500B_downstream_variant': 0,
         '5_prime_UTR_variant': 0,
         'frameshift_variant': 1,
         'intron_variant': 0,
         'missense_variant': 1,
         'nonsense': 1,
         'splice_acceptor_variant': 2,
         'splice_donor_variant': 2,
         'synonymous_variant': 1}
    else:
        raise ValueError("legendDict can be either effect_to_coding_legend or mc_to_coding_legend") 
    
    type_key = {0: 'non_coding',
           1: 'coding',
           2: 'splicing',
           3: 'ncRNA'}
    
    if x == x:
        recode = [legendDict[i] for i in x]
        if 1 in recode:
            return type_key[1]
        elif 2 in recode:
            return type_key[2]
        elif 3 in recode:
            return type_key[3]
        else:
            return type_key[0]
    else:
        return np.nan

def determine_final_coding_type(x):
    """
    This function is to decide the final coding type of variant if both
    ensemble effect and molecuar consequence effect exist.
    
    This function is to apply to each row.
    """

    e, m = x['iscoding_from_effect'], x['iscoding_from_mc']
    valid_values = ['non_coding','coding','splicing']
    
    if e == m:
        return e
    else:
        if (e in valid_values) and (m in valid_values):
            return 'inconsistent'
        elif e in valid_values:
            return e
        elif m in valid_values:
            return m
        else:
            return np.nan

def add_final_coding_type(df):
    if 'effect' in df.columns:
        df['iscoding_from_effect'] = df.apply(lambda x: determine_coding_from_legendDict(x['effect'], 'effect_to_coding_legend') , axis = 1)
    if 'molecular_consequence_recode' in df.columns:
        df['iscoding_from_mc'] = df.apply(lambda x: determine_coding_from_legendDict(x['molecular_consequence_recode'], 'mc_to_coding_legend') , axis = 1)

    if ('effect' in df.columns) and ('molecular_consequence_recode' in df.columns):
        df['final_coding_type'] = df.apply(lambda x: determine_final_coding_type(x), axis = 1)
    else:
        df['final_coding_type'] = df['iscoding_from_effect']


##############################################################################
## This is a high-level function to process the data using lower lever command
##############################################################################

def process_annotation(df):
    recode_annotation(df)
    add_final_coding_type(df)



#################
## Some variables
#################

contScore = ['deepsea_mlog2', 'phyloP46way_primate','phyloP46way_placental', 'phyloP100way_vertebrate', 
            'phastCons46way_primate','phastCons46way_placental', 'phastCons100way_vertebrate',
            'GERP_RS','SiPhy_29way_logOdds','integrated_fitCons_score','GenoCanyon_score',
            'CADD_raw','DANN_score','fathmm-MKL_non-coding_score','fathmm-MKL_coding_score','Eigen-PC-raw',
            'funseq2_noncoding_score']

sixteenScore = ['deepsea_mlog2', 'phyloP46way_primate','phyloP46way_placental', 'phyloP100way_vertebrate',
            'phastCons46way_primate','phastCons46way_placental', 'phastCons100way_vertebrate',
            'GERP_RS','SiPhy_29way_logOdds','integrated_fitCons_score','GenoCanyon_score',
            'CADD_raw','DANN_score','fathmm-MKL_non-coding_score','fathmm-MKL_coding_score','Eigen-PC-raw']

contRankScore = ['phyloP46way_primate_rankscore', 'phyloP46way_placental_rankscore',
                'phyloP100way_vertebrate_rankscore','phastCons46way_primate_rankscore',
                'phastCons46way_placental_rankscore','phastCons100way_vertebrate_rankscore',
                'GERP_RS_rankscore','SiPhy_29way_logOdds_rankscore','integrated_fitCons_rankscore',
                'GenoCanyon_rankscore','CADD_raw_rankscore','DANN_rank_score',
                 'fathmm-MKL_non-coding_rankscore','fathmm-MKL_coding_rankscore',
                'funseq2_noncoding_rankscore']

funcScore = ['deepsea_mlog2', 'integrated_fitCons_score','GenoCanyon_score','CADD_raw',
 'DANN_score','fathmm-MKL_non-coding_score','fathmm-MKL_coding_score','Eigen-PC-raw']



