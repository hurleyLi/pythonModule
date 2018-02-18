import pandas as pd


#############################################
## extended functions for variant annotations
#############################################

def variant_type(ref,alt):
    lref, lalt = len(ref), len(alt)
    if lref == lalt:
        if lref == 1:
            return 'SNV'
        else:
            return 'MNT_change'
    if lalt > lref:
            return 'insertion'
    if lalt < lref:
            return 'deletion'

def add_variant_type(df, ref = 'ref', alt = 'alt', new_column = 'variant_type', inplace = False):
    """
    Add varaint type by comparing the length of ref allele vs alt allele
    possible output: SNV, deletion, insertion, MNT_change
    
    -----------
    Parameters:
    ref : column name of reference allele
    alt : column name of alternative allele
    new_column : name of the new column containing the variant type
    inplace : if inplace == True, will not return anything
    """

    # make sure df have ref and alt columns
    if not inplace:
        d = df.copy()
    else:
        d = df
    
    if {ref, alt}.issubset(set(d.columns)):
        d[new_column] = d.apply(lambda x: variant_type(x['ref'],x['alt']),axis = 1)
    else:
        raise ValueError('dataframe does not contain columns %s and %s' % (ref, alt))
    
    if not inplace:
        return d

def clean_chr(df, chrCol = 'chr', keepX = True, keepY = True, inplace = False, exclude = []):
    """
    Remove chr other than chr1-22,X,Y. By default keep chr1-22,X,Y
    
    -----------
    Parameters:
    chrCol : column name containing the chr
    keepX : whether keep X
    keepY : whether keep Y
    inplace : if inplace == True, will not return anything
    """
    if chrCol not in df.columns:
        raise ValueError('dataframe does not contain columns %s' % chrCol)
    
    keepChr = [str(x) for x in range(23)]
    if keepX:
        keepChr.append('X')
    if keepY:
        keepChr.append('Y')
    if exclude:
        for c in exclude:
            if c in keepChr:
                keepChr.remove(c)
    
    if not inplace:
        d = df.copy()
    else:
        d = df
        
    idx = d[[x not in keepChr for x in d[chrCol]]].index
    d.drop(idx, inplace = True)
    
    if not inplace:
        return d


##################################
## General pd dataframe extensions
##################################

def select_row(df, criteria_dict):
    """
    pd.DataFrame extension for selecting columns based on criteria in dictionary.
    
    The key in dictionary is column name or index. If the value is a list,
    this function will match the column with the values in the list
    """
    
    names = df.columns
    
    for key, item in criteria_dict.items():
        if key in names:
            if isinstance(item, list):
                df = select_row_by_matching_list(df, key, item)
            else:
                df = select_row_by_value(df, key, item)
        else:
            raise ValueError("key name %s not in the header of dataframe" % key)
    return df

## variant selection functions
def select_row_by_value(df, colname, value):
    return df[df[colname] == value]
def select_row_by_matching_list(df, colname, match_list):
    return df[[ x in match_list for x in df[colname]]]



setattr(pd.DataFrame, "add_variant_type", add_variant_type)
setattr(pd.DataFrame, "clean_chr", clean_chr)
setattr(pd.DataFrame, "select_row", select_row)



