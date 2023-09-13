import numpy as np
import matplotlib.pyplot as plt
import colorspacious as cs

rgb2lab = cs.cspace_converter("sRGB1", "CAM02-UCS")

def frmt(x:np.ndarray, precision=2):
    '''
    Formats a numpy array to have floats with `precision` digits

    Parameters
    ----------
    x : np.ndarray
        array to be formatted
    precision : int, optional
        number of digits, by default 2

    Returns
    -------
    np.ndarray
        formatted array
    '''
    _x = x.flatten()
    _x = np.array([f'{v:.{precision}f}' for v in _x])
    return _x.reshape(x.shape)

def table(vals, col_labels, row_labels, norm=None, vmin=None, vmax=None, color_range=None, cmap=plt.cm.hot, text_digits=2,
          num=None, figsize=(7,3), xlabel=None, ylabel=None, title=None, label_fontsize=12, title_fontsize=14):
    """
    Generate a table plot with cells colored according to their value
    
    Parameters:
    - vals: numpy.ndarray
        The values to be plotted in the table. It should have shape (len(row_labels), len(col_labels)).
    - col_labels: list
        The labels for the columns of the table.
    - row_labels: list
        The labels for the rows of the table.
    - norm: matplotlib.colors.Normalize, optional
        The normalization object used to scale the values for coloring the cells. If not provided, the minimum and maximum values
        of the data will be used to create a default normalization. Alternatively, you can provide vmin and vmax explicitly.
    - vmin: float, optional
        The minimum value of the data. If not provided, the minimum value of the data will be used.
    - vmax: float, optional
        The maximum value of the data. If not provided, the maximum value of the data will be used.
    - cmap: matplotlib.colors.Colormap, optional
        The colormap used to map the values to colors. The default colormap is 'hot'.
    - text_digits: int, optional
        The number of digits to display in the table cells.
    - num: int, optional
        The number of the figure. If provided, the figure will be closed before creating a new one.
    - figsize: tuple, optional
        The size of the figure in inches. The default size is (7, 3).
    - xlabel: str, optional
        The label for the x-axis of the plot.
    - ylabel: str, optional
        The label for the y-axis of the plot.
    - title: str, optional
        The title of the plot.
    - label_fontsize: int, optional
        The font size of the axis labels.
    - title_fontsize: int, optional
        The font size of the plot title.
        
    Returns:
    - matplotlib.figure.Figure
        The generated table plot figure.
    """
    assert vals.shape == (len(row_labels), len(col_labels))
    
    if num:
        plt.close(num)
    fig = plt.figure(num=num, figsize=figsize)
    ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])
    
    # properly define norm
    if norm is None:
        if vmin is None:
            vmin = np.nanmin(vals)
        if vmax is None:
            vmax = np.nanmax(vals)
        if color_range is not None:
            assert len(color_range) == 2
            assert color_range[0] < color_range[1]
            vmin, vmax = (np.array([0,1]) - color_range[0]) * (vmax - vmin) / (color_range[1] - color_range[0]) + vmin
        norm = plt.Normalize(vmin, vmax)
    colours = cmap(norm(vals))

    the_table=plt.table(cellText=frmt(vals,text_digits), rowLabels=row_labels, colLabels=col_labels, 
                        colWidths = [1/(vals.shape[1] + 1)]*vals.shape[1],
                        loc='center', 
                        cellColours=colours,
                       )
    the_table.scale(1,2)
    ax.xaxis.set_label_position('top')
    plt.xlabel(xlabel,fontsize=label_fontsize)
    plt.ylabel(ylabel,fontsize=label_fontsize)
    plt.title(title, fontsize=title_fontsize)

    fig.tight_layout()
    
    return fig

def tex_table(vals, col_labels, row_labels, norm=None, vmin=None, vmax=None, color_range=None, cmap=plt.cm.hot, text_digits=2, rgb_digits=8, white_text_if_lightness_below=10, xlabel=None, ylabel=None, title=None, side_xlabel=False, close_left=True, close_top=True):
    """
    Generates a LaTeX table coloring the cells based on their values.

    Add to your preamble the following:
    \\usepackage{graphicx,multirow}
    \\usepackage[table]{xcolor}
    
    Args:
        vals (ndarray): A 2D array of values for the table. It should have shape (len(row_labels), len(col_labels)).
        col_labels (list): A list of column labels.
        row_labels (list): A list of row labels.
        norm (Normalize, optional): A normalization object to normalize the values. Defaults to None.
        vmin (float, optional): The minimum value for the normalization. Defaults to None.
        vmax (float, optional): The maximum value for the normalization. Defaults to None.
        cmap (Colormap, optional): The colormap to use for coloring the table cells. Defaults to plt.cm.hot.
        text_digits (int, optional): The number of digits to display for the values in the table cells. Defaults to 2.
        rgb_digits (int, optional): The number of digits to display for the RGB values of the cell colors. Defaults to 8.
        xlabel (str, optional): The label for the x-axis. Defaults to None.
        ylabel (str, optional): The label for the y-axis. Defaults to None.
        title (str, optional): The title of the table. Defaults to None.
        side_xlabel (bool, optional): Whether to display the xlabel on the side instead of on top. Defaults to False.
        close_left (bool, optional): Whether to close the left border of the table. Defaults to True.
        close_top (bool, optional): Whether to close the top border of the table. Defaults to True.
    
    Returns:
        str: The LaTeX code for the generated table, as string.
    """
    assert vals.shape == (len(row_labels), len(col_labels))

    # properly define norm
    if norm is None:
        if vmin is None:
            vmin = np.nanmin(vals)
        if vmax is None:
            vmax = np.nanmax(vals)
        if color_range is not None:
            assert len(color_range) == 2
            assert color_range[0] < color_range[1]
            vmin, vmax = (np.array([0,1]) - color_range[0]) * (vmax - vmin) / (color_range[1] - color_range[0]) + vmin
        norm = plt.Normalize(vmin, vmax)
        
    colours = cmap(norm(vals))
    
    if side_xlabel and xlabel:
        side_xlabel = xlabel
        xlabel = None
    else:
        side_xlabel = ''
    
    tbl = "\\begin{tabular}{%s}\n" %(('|' if close_left else '') + 'c|'*(len(col_labels) + 1 + bool(ylabel)))
    if title:
        tbl += "\t\multicolumn{%d}{c}{%s} \\\\\n" %(len(col_labels) + 1 + bool(ylabel), title)
    if close_top or not xlabel:
        tbl += "\t\cline{%d-%d}\n" %(2 + bool(ylabel), len(col_labels) + 1 + bool(ylabel))
    elif title:
        tbl += "\t\midrule\n"
    
    # xlabel top line
    if xlabel:
        if ylabel:
            tbl += "\t\multicolumn{2}{%s}{} & " %('c|' if close_top else 'c')
        else:
            tbl += "\t & "
        tbl += "\multicolumn{%d}{%s}{%s} \\\\\n" %(len(col_labels), 'c|' if close_top else 'c', xlabel)
        tbl += "\t\cline{%d-%d}\n" %(2 + bool(ylabel), len(col_labels) + 1 + bool(ylabel))
    
    # row labels
    if ylabel:
        tbl += "\t\multicolumn{2}{c|}{%s} & " %(side_xlabel)
    else:
        tbl += "\t%s & " %(side_xlabel)
    tbl += ' & '.join(str(cl) for cl in col_labels) + ' \\\\\n'
    tbl += "\t\cline{%d-%d}\n" %(bool(ylabel) + 1 - bool(close_left), len(col_labels) + 1 + bool(ylabel))
    
    # ylabel leftmost column
    if ylabel:
        tbl += "\t\multirow{%d}{*}{\\rotatebox[origin=c]{90}{%s}}\n" %(len(row_labels), ylabel)
    
    # values
    for r in range(len(row_labels)):
        tbl += '\t'
        if ylabel:
            tbl += ' & '
        tbl += f'{row_labels[r]}'
        for c in range(len(col_labels)):
            v = vals[r,c]
            rgb = ''
            if np.isnan(v):
                v = '-'
                rgb = '1,1,1' # make the cell white
            else:
                v = f'{v:.{text_digits}f}'
                rgb = colours[r,c,:3] # get rid of the alpha parameter
                if white_text_if_lightness_below and rgb2lab(rgb)[0] < white_text_if_lightness_below:
                    v = '\\textcolor{white}{%s}' %v
                rgb = ','.join([f'{_rgb:.{rgb_digits}f}' for _rgb in rgb])
            tbl += " & \cellcolor[rgb]{" + rgb + '}' + v
        tbl += ' \\\\\n'
        if r < len(row_labels) - 1:
            tbl += "\t\cline{%d-%d}\n" %(1 + bool(ylabel), len(col_labels) + 1 + bool(ylabel))
        else:
            tbl += "\t\cline{%d-%d}\n" %(1 - bool(close_left) + bool(ylabel), len(col_labels) + 1 + bool(ylabel))
    
    tbl += "\\end{tabular}"
    
    return tbl