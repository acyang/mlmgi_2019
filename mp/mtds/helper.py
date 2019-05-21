
import numpy as np
from numpy import zeros, mean
from pymatgen import Element, Composition


def filter_generic(mt_list, func):
    """Return a list of materials filtered by func.
    
    Args:
    mt_list: a list of materials.
    func: a function apply to mt and returns a boolean.
    """
    return [mt for mt in mt_list if func(mt)]

def filter_element_and(mt_list, elem_list):
    """ Return a list of materials with ALL elements(AND) in elem_list.
    
    Args:
    mt_list: a list of materials.
    elem_list: a list of elements, e.g. ['Si', 'O']
    """
    return [mt for mt in mt_list if all(e in mt['pretty_formula'] for e in elem_list)]

def filter_element_or(mt_list, elem_list):
    """ Return a list of materials with ANY elements(OR) in elem_list.
    
    Args:
    mt_list: a list of materials.
    elem_list: a list of elements, e.g. ['Si', 'O']
    """
    return [mt for mt in mt_list if any(e in mt['pretty_formula'] for e in elem_list)]

def random_sampling(elements, n):
    """ Return a sub-list of n samples from a list."""
    import random
    return [random.choice(elements) for i in range(n)]

#def checkRowGroupBound():
def row_group_limits():
    """Return the maximum # of row and group of the periodic table."""
    from pymatgen import Element, periodic_table
    
    # Get all available elements in periodic table.
    rs = [e.row for e in periodic_table.Element]
    gs = [e.group for e in periodic_table.Element]
    
    return (max(rs), max(gs))
	
def get_row_group_density(mt, rg_limits):
    """Return the atomic fraction corresponding to row and group position."""
    
    fraction_matrix = zeros(rg_limits)
    
    composition = Composition(mt['pretty_formula'])
    
    for element in composition:
        fraction = composition.get_atomic_fraction(element)
        elem  = Element(element)
        row   = elem.row
        group = elem.group
        fraction_matrix[row-1][group-1] = fraction
        
    return fraction_matrix
	
def get_element_density(mt):
    """Return a vector in which the elements are corresponding to its atomic fraction.
    """
    fraction_matrix = zeros(100)
    
    composition = Composition(mt['pretty_formula'])
    
    for element in composition:
        fraction = composition.get_atomic_fraction(element) # get the atomic fraction.
        fraction_matrix[element.Z] = fraction
        
    return fraction_matrix

def get_row_group_density_vec(mt, rg_limits):
    """Return the atomic fraction according to atomic order(Z)."""
    rg_matrix = get_row_group_density(mt, rg_limits)
    rd_vec = np.sum(rg_matrix, axis=1)
    gd_vec = np.sum(rg_matrix, axis=0)
    
    return np.concatenate((rd_vec, gd_vec))
	
	
def check_elasticity(mt, threshold = 0.1):
    """Check if the elasticity is available."""
	
    if mt['elasticity'] == None:
        return False
    elif mt['elasticity']['elastic_tensor'][0][0] < threshold:  # This is a temporary implement.
        return False
    else:
        return True
		
def aggregate_data(mts, feature, target):
    """Return a dictionary contains corresponding materials, features and targets.
	"""
    set_dict = dict()
    set_dict['mt'] = mts
    set_dict['feature'] = feature
    set_dict['target']  = target
    
    return set_dict
		
def build_feature_target_pair(mts, fun_get_features, fun_get_targets):
    """Return paired features and targets.
    
    Args:
    mts: a list of materials.
    funGetFeatures: a function to extract the feature(s)
    funGetTargets : a function to get the target value(s)
    """
    features = []
    targets  = []
    for mt in mts:
        targets.append(fun_get_targets(mt))
        features.append(fun_get_features(mt))
    
    return features, targets
	
def show_correlations(regressors, features, targets):
    """Show the MAE & pearson's correlation of predictions of regressors.
    
    Args:
    regressors -- a list of dictionaries.
				  each dictionary contains 'name':string, 'regressor':scikit-learn estimator.
    """
    
    def mae(v1, v2):
    #"""Return the MAE (mean absolute error) of v1 & v2."""
        return mean(abs(v1 - v2))
	
    from sklearn.metrics import matthews_corrcoef
    from scipy.stats import pearsonr
	
    for regressor in regressors:
        regressor['preds'] = regressor['regressor'].predict(features)
        
    print('=============== MAE Comparison =================')
    for regressor in regressors:
        print('{} : {}'.format(regressor['name'], mae(regressor['preds'], targets)))
        
    print("=============== Pearson's Correlation Comparison =================")
    for regressor in regressors:
        print('{} : {}'.format(regressor['name'], pearsonr(regressor['preds'], targets)))
