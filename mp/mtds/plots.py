
from numpy import zeros, mean
from pymatgen import Element, Composition

def row_group_limits():
    """Return the maximum # of row and group of a periodic table."""
    from pymatgen import Element, periodic_table
    
    # Get all available elements in periodic table.
    rs = [e.row for e in periodic_table.Element]
    gs = [e.group for e in periodic_table.Element]
    
    return (max(rs), max(gs))

def show_elements_on_ptable(mt_groups):
    """Shows the # of occurrence of each element on periodic table. 
	"""
    from pymatgen import periodic_table
    import plotly.figure_factory as ff
    
    num_of_groups = len(mt_groups)
    
    # Get the limits of periodic table.
    rg_lims = row_group_limits()
    bin_groups = [zeros(rg_lims, dtype = int) for i in range(num_of_groups)]
	
	# Print the # of elements in the groups.
    for i in range(num_of_groups):
        print('Total # of materials in Group {}: {}'.format(i, len(mt_groups[i])))
    
	# Accumulate the element appearance frequency.
    for i in range(num_of_groups):
        for mt in mt_groups[i]:
            composition = Composition(mt['pretty_formula'])
            for e in composition:
                r, g = e.row - 1, e.group - 1
                bin_groups[i][r][g] += 1
        
    # Prepare an empty list. 
    pt_show = [['' for i in range(rg_lims[1] + 1)] for j in range(rg_lims[0] + 1)]
	
    # Insert the headers of rows & columns.
    pt_show[0][1:] = [ 1 + j for j in range(rg_lims[1])]
    for j in range(rg_lims[0]): pt_show[1 + j][0] = 1 + j 
    
    # Combine Element info & frequency info.
    for e in periodic_table.Element:
        r, g = e.row, e.group
        msg = '<br>'.join(str(bin_groups[i][r - 1][g - 1]) for i in range(num_of_groups))
        pt_show[r][g] = str(e) + '<br>' + msg 
            
    from plotly.offline import plot, iplot, init_notebook_mode
    import plotly.graph_objs as go
    
    # This line is necessary for offline mode.
    init_notebook_mode(connected = False)
    
    table = ff.create_table(pt_show)
    table.layout.height = 30 + 300*num_of_groups
    
    print('==========================================================')
    iplot(table)
    
def mt_group4plot(mts, name, marker):
    """Return a dict as input(s) for plotly_properties."""
    g = dict()
    g['mts'] = mts
    g['name'] = name
    g['marker'] = marker
    return g
	
def plotly_properties(mt_groups, get_x=lambda x:x['density'], get_y=lambda x:x['volume']):
    """Show a 2-D plot of feature1 respect to feature2 of mt_groups. """
    from plotly.offline import plot, iplot, init_notebook_mode
    import plotly.graph_objs as go

    # This line is necessary for offline mode.
    init_notebook_mode(connected=False)
    
    scatters = []
    
    for mts in mt_groups:
        tmp_scatter = go.Scatter(
        x = [get_x(mt) for mt in mts['mts']],
        y = [get_y(mt) for mt in mts['mts']],
        mode = 'markers',
        text = ['{}|{}'.format(mt['pretty_formula'], mt['full_formula']) for mt in mts['mts']],
        name = mts['name'],
        marker = mts['marker']
        )
        
        scatters.append(tmp_scatter)
    
    data = go.Data(scatters)
   
    x_label = get_x.__name__
    y_label = get_y.__name__
    layout=go.Layout(title="Plot [{}] v.s. [{}]".format(x_label, y_label), 
                     xaxis={'title':x_label}, yaxis={'title':y_label})
    
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)
    
def data4plotly(training_set, testing_set, predictor):
    """Return python dicts(of training and testing set) for further plotly processing.
	"""
    
    training_dict = dict()
    testing_dict  = dict()
    
    # Put the training set in to proper format.
    training_dict['real']    = training_set['target']
    training_dict['predict'] = predictor.predict(training_set['feature'])
    training_dict['tag']     = ["Pretty:{}, Full:{}".format(mt['pretty_formula'], mt['full_formula']) for mt in training_set['mt']]

    # Put the training set in to proper format.
    testing_dict['real']    = testing_set['target']
    testing_dict['predict'] = predictor.predict(testing_set['feature'])
    testing_dict['tag']     = ["Pretty:{}, Full:{}".format(mt['pretty_formula'], mt['full_formula']) for mt in testing_set['mt']]
    
    return training_dict, testing_dict
	
def plotly_pairing_set(training, testing):
    """Plot the training & testing sets."""
    from plotly.offline import plot, iplot, init_notebook_mode
    import plotly.graph_objs as go

    # This line is necessary for offline mode.
    init_notebook_mode(connected=False)
    
    scatters = []
    
    training_scatter = go.Scatter(
        x = training['real'],
        y = training['predict'],
        mode = 'markers',
        text = training['tag'],
        name = 'Training Set',
        marker = dict(symbol='square-open', size=7, color='blue'),
    )
    
    testing_scatter = go.Scatter(
        x = testing['real'],
        y = testing['predict'],
        mode = 'markers',
        text = testing['tag'],
        name = 'Testing Set',
        marker = dict(symbol='circle-open', size=7, color='red'),
    )
    
    scatters.append(training_scatter)
    scatters.append(testing_scatter)
    
    # Draw a line with 1.0 correlation.
    end_pt = max(training['real'] + testing['real'])
    line = go.Scatter(x = [0, end_pt], y = [0, end_pt], mode = 'lines', name = 'r = 1.0')
    scatters.append(line)
    
    data = go.Data(scatters)
   
    x_label = 'real'
    y_label = 'predict'
    layout=go.Layout(title="Plot [{}] v.s. [{}]".format(x_label, y_label), 
                     xaxis={'title':x_label}, yaxis={'title':y_label})
    
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)
	
def plot_regression_results(training_set, testing_set, regressor, print_details = True):
    """Plot the regression results of training set, testing set.
	
	Args:
	regressor: a scikit-learn estimator.
	"""
    
    from sklearn.metrics import matthews_corrcoef
    from scipy.stats import pearsonr
	
    def mae(v1, v2):
    #"""Return the MAE (mean absolute error) of v1 & v2."""
        return mean(abs(v1 - v2))
    
#    predicts_train = training_set['target'] - regressor['regressor'].predict(training_set['feature'])
#    predicts_test  = testing_set['target']  - regressor['regressor'].predict(testing_set['feature'])
	
    training, testing = data4plotly(training_set, testing_set, regressor['regressor'])
	
    if print_details:
        print("{} Regressor".format(regressor['name']))
        print('Train - MAE: {}'.format(mae(training_set['target'], training['predict'])))
        print('Test  - MAE: {}'.format(mae(testing_set['target'], testing['predict'])))
        print('Train - Pearson r: {}'.format(pearsonr(training_set['target'], training['predict'])))
        print('Test  - Pearson r: {}'.format(pearsonr(testing_set['target'], testing['predict'])))
    
    training, testing = data4plotly(training_set, testing_set, regressor['regressor'])
    
    plotly_pairing_set(training, testing)
	
def plot_hist(list_of_data, plot_title, bin_sz):
    """Plot the overlay histogram of data sets.
    """
    
    from plotly.offline import plot, iplot, init_notebook_mode
    import plotly.graph_objs as go

    # This line is necessary for offline mode.
    init_notebook_mode(connected=False)
    
    data = []
    
    for d in list_of_data:
        
        trace_tmp = go.Histogram(
            x=d,
            opacity=0.33,
            autobinx=False,
            xbins=dict(start=min(d),end=max(d),size=bin_sz) 
        )
        
        data.append(trace_tmp)

    layout = go.Layout(title = plot_title, barmode='overlay')
    fig = go.Figure(data=data, layout=layout)

    iplot(fig, filename='Histograms')
	
def display_cv(scores):
    """Display the result of cross-validation pretty.
    """
    import tabulate
    import numpy as np
    
    scores = list(np.asarray(scores).T)
    print('====== {}-Fold Cross Validation ======'.format(len(scores)))
    headers = ['Model {}'.format(i) for i in range(len(scores))]
    
    print(tabulate.tabulate(scores, headers=headers))