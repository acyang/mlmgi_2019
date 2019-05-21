
import os

def read_vasp(file_path):
    """Return vasp as a dictionary.
	"""
    with open(file_path, 'r') as f:
        return eval(f.read())

# Read all VASP files in a directory.
def read_vasp_from_dir(path):
    """Return a list of dictionaries of vasp files.
    
    Args:
	  path: the directory where vasps located.
    """
    file_names = os.listdir(path)
    
    # Convert all the VASP files into a list of dictionaries.
    mts = []

    for file_name in file_names:
        file_path = os.path.join(path, file_name)
        mt = read_vasp(file_path)
        mts.append(mt)
    
    return mts

# For filtering materials of "material_id" with 'mp-'.
# Note: 'material_id' with 'mvc' prefix is annotataed with 
# 'gap value is approximate and using a loose k-point mesh'
def mt_filter(materials, token, item):
    """
	"""
    return [mt for mt in materials if token in mt[item]]