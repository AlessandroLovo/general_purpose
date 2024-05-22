import numpy as np
import matplotlib.pyplot as plt
import colorspacious as cs
from PIL import Image
import os
import pandas as pd

import uplotlib as uplt

def file_no_duplicate(filename:str):
    """
    Generates a unique filename by appending a numerical suffix if the given filename already exists.

    Parameters:
        filename (str): The original filename.

    Returns:
        str: A unique filename that does not exist in the current directory or its subdirectories.
    """
    if not os.path.exists(filename):
        return filename
    spl = filename.rsplit('/',1)
    if len(spl) == 2:
        path, filename = spl
    else:
        path = '.'
    spl  = filename.rsplit('.',1)
    if len(spl) == 2:
        filename, ext = spl
        ext = f'.{ext}'
    else:
        ext = ''
    uniq = 1
    while True:
        new_filename = f'{path}/{filename}_{uniq}{ext}'
        if not os.path.exists(new_filename):
            return new_filename
        uniq += 1

def fig2img(fig:plt.Figure, dpi=300, save_name=None, keep_filename=False):
    """
    Convert a matplotlib figure to a PIL Image, by saving the figure and then loading it as a PIL Image.

    Parameters:
        fig (plt.Figure): The matplotlib figure to convert.
        dpi (int, optional): The resolution in dots per inch. Defaults to 300.
        save_name (str, optional): The name of the file to save the figure as. If not provided, a temporary file name will be used. Defaults to None.
        keep_filename (bool, optional): Whether to keep the saved file after converting. Defaults to False.

    Returns:
        im (Image): The converted PIL Image.
    """
    if save_name is None:
        save_name = 'temp.png'
    save_name = file_no_duplicate(save_name)
    fig.savefig(save_name, dpi=dpi)
    im = Image.open(save_name)
    if not keep_filename:
        os.remove(save_name)
    return im

rgb2lab = cs.cspace_converter("sRGB1", "CAM02-UCS")

def rescale(image:Image.Image, scale_factor:float=1):
    """
    Rescales the given image by a specified scale factor.

    Parameters:
        image (PIL.Image.Image): The image to be rescaled.
        scale_factor (float): The factor by which the image should be rescaled. Default is 1.

    Returns:
        PIL.Image.Image: The rescaled image.
    """
    new_size = np.array(image.size) * scale_factor
    new_size_int = tuple([int(l) for l in new_size])
    return image.resize(new_size_int)

def table(vals, col_labels=None, row_labels=None, norm=None, vmin=None, vmax=None, color_range=None, cmap='hot', text_digits=2, ufloat_digits=1,
          num=None, figsize=(7,3), xlabel=None, ylabel=None, title=None, label_fontsize=12, title_fontsize=14):
    """
    Generate a table plot with cells colored according to their value
    
    Parameters:
    - vals: numpy.ndarray or pandas.DataFrame
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
    - ufloat_digits: int, optional
        The number of significant digits to display in the table cells for ufloat values.
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
    # deal with vals as DataFrame
    if isinstance(vals, pd.DataFrame):
        vals = vals.values
        if col_labels is None:
            col_labels = vals.columns
        if row_labels is None:
            row_labels = vals.index
    
    if col_labels is None:
        col_labels = list(range(vals.shape[1]))
    if row_labels is None:
        row_labels = list(range(vals.shape[0]))
    
    assert vals.shape == (len(row_labels), len(col_labels))
    cmap = plt.get_cmap(cmap)
    
    if num:
        plt.close(num)
    fig = plt.figure(num=num, figsize=figsize)
    ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])
    
    _vals = uplt.nominal_value(vals)

    # properly define norm
    if norm is None:
        if vmin is None:
            vmin = np.nanmin(_vals)
        if vmax is None:
            vmax = np.nanmax(_vals)
        if color_range is not None:
            assert len(color_range) == 2
            assert color_range[0] < color_range[1]
            vmin, vmax = (np.array([0,1]) - color_range[0]) * (vmax - vmin) / (color_range[1] - color_range[0]) + vmin
        norm = plt.Normalize(vmin, vmax)
    colours = cmap(norm(_vals))

    the_table=plt.table(cellText=uplt.vectorized_frmt(vals,text_digits,ufloat_digits), rowLabels=row_labels, colLabels=col_labels, 
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

def tex_table(vals, col_labels=None, row_labels=None, norm=None, vmin=None, vmax=None, color_range=None, cmap='hot', text_digits=2, error_digits=1, rgb_digits=8, white_text_if_lightness_below=25, xlabel=None, ylabel=None, title=None, center_title=False, side_xlabel=False, close_left=True, close_top=True, use_midrule=True, leading_indentation=0):
    """
    Generates a LaTeX table coloring the cells based on their values.

    Add to your preamble the following:
    \\usepackage{graphicx,multirow}
    \\usepackage[table]{xcolor}
    
    Args:
        vals (ndarray|pd.DataFrame): A 2D array of values for the table. It should have shape (len(row_labels), len(col_labels)).
        col_labels (list): A list of column labels.
        row_labels (list): A list of row labels.
        norm (Normalize, optional): A normalization object to normalize the values. Defaults to None.
        vmin (float, optional): The minimum value for the normalization. Defaults to None.
        vmax (float, optional): The maximum value for the normalization. Defaults to None.
        cmap (Colormap, optional): The colormap to use for coloring the table cells. Defaults to plt.cm.hot.
        text_digits (int, optional): The number of digits to display for the values in the table cells. Defaults to 2.
        error_digits (int, optional): The number of significan digits of the error in case the values are ufloats. Defaults to 1.
        rgb_digits (int, optional): The number of digits to display for the RGB values of the cell colors. Defaults to 8.
        xlabel (str, optional): The label for the x-axis. Defaults to None.
        ylabel (str, optional): The label for the y-axis. Defaults to None.
        title (str, optional): The title of the table. Defaults to None.
        side_xlabel (bool, optional): Whether to display the xlabel on the side instead of on top. Defaults to False.
        close_left (bool, optional): Whether to close the left border of the table. Defaults to True.
        close_top (bool, optional): Whether to close the top border of the table. Defaults to True.
        use_midrule (bool, optional): Whether to use midrule or hline in the table. Defaults to True (midrule).
        leading_indentation (int, optional): The indentation of the table, for easier copy-pasting. Defaults to 0.
    
    Returns:
        str: The LaTeX code for the generated table, as string.
    """
    leading_indentation = '\t' * leading_indentation

    # deal with vals as DataFrame
    if isinstance(vals, pd.DataFrame):
        vals = vals.values
        if col_labels is None:
            col_labels = vals.columns
        if row_labels is None:
            row_labels = vals.index

    if row_labels is not None:
        assert vals.shape[0] == len(row_labels)
    else:
        assert ylabel is None, 'If row_labels is None, ylabel must be None'
    if col_labels is not None:
        assert vals.shape[1] == len(col_labels)
    else:
        assert xlabel is None, 'If col_labels is None, xlabel must be None'
    nrow, ncol = vals.shape

    extra_left_cols = bool(ylabel) + (row_labels is not None)

    if cmap is not None:
        cmap = plt.get_cmap(cmap)

        _vals = uplt.nominal_value(vals)

        # properly define norm
        if norm is None:
            if vmin is None:
                vmin = np.nanmin(_vals)
            if vmax is None:
                vmax = np.nanmax(_vals)
            if color_range is not None:
                assert len(color_range) == 2
                assert color_range[0] < color_range[1]
                vmin, vmax = (np.array([0,1]) - color_range[0]) * (vmax - vmin) / (color_range[1] - color_range[0]) + vmin
            norm = plt.Normalize(vmin, vmax)
            
        colours = cmap(norm(_vals))
    
    if side_xlabel and xlabel and row_labels is not None:
        side_xlabel = xlabel
        xlabel = None
    else:
        side_xlabel = ''
    
    tbl = leading_indentation + "\\begin{tabular}{%s}\n" %(('|' if close_left else '') + 'c|'*(ncol + extra_left_cols))
    if title:
        tbl += leading_indentation
        if center_title:
            tbl += "\t\multicolumn{%d}{c}{%s} \\\\\n" %(ncol + extra_left_cols, title)
        elif extra_left_cols:
            tbl += "\t\multicolumn{%d}{c}{} & \multicolumn{%d}{c}{%s} \\\\\n" %(extra_left_cols, ncol, title)
        else:
            tbl += "\t\multicolumn{%d}{c}{%s} \\\\\n" %(ncol, title)
    if close_top or not xlabel:
        tbl += leading_indentation +"\t\cline{%d-%d}\n" %(1 + extra_left_cols, ncol + extra_left_cols)
    elif title:
        tbl += leading_indentation + "\t%s\n" %('\midrule' if use_midrule else '\hline')
    
    # xlabel top line
    if xlabel:
        tbl += leading_indentation
        if extra_left_cols:
            tbl += "\t\multicolumn{%d}{%s}{} & " %(extra_left_cols, 'c|' if close_top else 'c')
        else:
            tbl += "\t"
        tbl += "\multicolumn{%d}{%s}{%s} \\\\\n" %(ncol, ('|' if close_left and not extra_left_cols else '') + ('c|' if close_top else 'c'), xlabel)
        tbl += leading_indentation + "\t\cline{%d-%d}\n" %(1 + extra_left_cols, ncol + extra_left_cols)
    
    # col labels
    if col_labels is not None:
        tbl += leading_indentation
        if extra_left_cols:
            tbl += "\t\multicolumn{%d}{c|}{%s} & " %(extra_left_cols, side_xlabel)
        else:
            tbl += "\t"
        tbl += ' & '.join(str(cl) for cl in col_labels) + ' \\\\\n'
        tbl += leading_indentation + "\t\cline{%d-%d}\n" %(max(1, extra_left_cols - bool(close_left)), ncol + extra_left_cols)
    
    # ylabel leftmost column
    if ylabel:
        tbl += leading_indentation + "\t\multirow{%d}{*}{\\rotatebox[origin=c]{90}{%s}}\n" %(nrow, ylabel)
    
    # values
    for r in range(nrow):
        tbl += leading_indentation + '\t'
        if ylabel:
            tbl += ' & '
        if row_labels is not None:
            tbl += f'{row_labels[r]} &'
        for c in range(ncol):
            if c > 0:
                tbl += ' & '
            v = vals[r,c]

            # format value
            if np.isnan(v):
                v = '-'
            else:
                v = uplt.frmt(v, text_digits, error_digits)

            # deal with color
            if cmap is not None:
                if v == '-':
                    rgb = '1,1,1' # make the cell white
                else:
                    rgb = colours[r,c,:3] # get rid of the alpha parameter
                    if white_text_if_lightness_below and rgb2lab(rgb)[0] < white_text_if_lightness_below:
                        v = '\\textcolor{white}{%s}' %v
                    rgb = ','.join([f'{_rgb:.{rgb_digits}f}' for _rgb in rgb])
                tbl += "\cellcolor[rgb]{" + rgb + '}'

            tbl += v
        tbl += ' \\\\\n'
        if r < nrow - 1:
            tbl += "\t\cline{%d-%d}\n" %(max(1,extra_left_cols), ncol + extra_left_cols)
        else:
            tbl += "\t\cline{%d-%d}\n" %(max(1,extra_left_cols - bool(close_left)), ncol + extra_left_cols)
    
    tbl += leading_indentation + "\\end{tabular}"
    
    return tbl


def make_label_fig(text, figsize, rotation=0):
    """
    Generate a figure with a labeled text.

    Parameters:
    - text (str): The text to be displayed on the figure.
    - figsize (tuple): The size of the figure in inches (width, height).
    - rotation (int, optional): The rotation angle of the text in degrees. Default is 0.

    Returns:
    - fig (Figure): The generated figure object.
    """
    plt.close(0)
    fig = plt.figure(figsize=figsize, num=0)
    fig.text(0.5,0.5,text, ha='center',va='center', rotation=rotation)
    return fig

def fig_table_from_ims(ims, col_label_ims=None, row_label_ims=None, h_spacing=0, v_spacing=None):
    """
    Generates a figure table from a given array of images.

    Args:
        ims (ndarray): Array of Image.Image objects.
        col_label_ims (ndarray, optional): Array of Image.Image objects for column labels. Defaults to None.
        row_label_ims (ndarray, optional): Array of Image.Image objects for row labels. Defaults to None.
        h_spacing (int, optional): Horizontal spacing between images. Defaults to 0.
        v_spacing (int, optional): Vertical spacing between images. Defaults to None.
            For both h_spacing and v_spacing, if one is None, the other is set to the same value. Numerical values are relative to the size of the images in the table.
            For example h_spacing = 0.1, means the horizontal spacing will be one tenth of the width of the widest image in `ims`.

    Returns:
        Image: The generated figure table.
    """
    if col_label_ims is not None:
        assert ims.shape[1] == len(col_label_ims)
        col_label_ims = np.array(col_label_ims, dtype=object)
    if row_label_ims is not None:
        assert ims.shape[0] == len(row_label_ims)
        row_label_ims = np.array(row_label_ims, dtype=object)
        
    if col_label_ims is not None:
        ims = np.vstack([col_label_ims, ims])
        if row_label_ims is not None:
            cornerpiece = Image.new('RGB', (2,2))
            row_label_ims = np.array([cornerpiece] + list(row_label_ims), dtype=object)
    if row_label_ims is not None:
        ims = np.vstack([row_label_ims, ims.T]).T

    nrow, ncol = ims.shape
        
    sizes = np.array([im.size for im in ims.reshape(-1)]).reshape((nrow,ncol,2))
    col_widths = np.max(sizes[...,0], axis=0)
    row_heights = np.max(sizes[...,1], axis=1)
    if h_spacing is not None:
        h_spacing = int(h_spacing*np.max(col_widths))
        if v_spacing is None:
            v_spacing = h_spacing
        else:
            v_spacing = int(v_spacing*np.max(row_heights))
    elif v_spacing is not None:
        h_spacing = v_spacing = int(v_spacing*np.max(row_heights))
    else:
        h_spacing = v_spacing = 0
    
    core_height = np.sum(row_heights) + v_spacing*(nrow - 1)
    core_width = np.sum(col_widths) + h_spacing*(ncol - 1)
    
    new_im_size = (core_width, core_height)
    
    new_im = Image.new('RGB', new_im_size)
    
    
    anchor = np.array([0,0], dtype=int)
    for r in range(nrow):
        h = row_heights[r] # cell height
        anchor[0] = 0
        for c in range(ncol):
            w = col_widths[c] # col width
            im = ims[r,c]
            offset = np.array((w - im.size[0])//2, (h - im.size[1])//2)
            
            new_im.paste(im, tuple(anchor + offset))
            
            anchor[0] += w + h_spacing
        anchor[1] += h + v_spacing
            
    return new_im

def fig_table(plotting_function, vals:np.ndarray, col_labels=None, row_labels=None, figsize=(7,3), dpi=300, label_thickness=1, h_spacing=0, v_spacing=None, **kwargs):
    """
    Generate a table of figures using the given plotting function and values.

    Parameters:
        - plotting_function: The function used to generate each individual figure, according to the values in `vals`.
            The signature of the function should be:
                plotting_function(v, figsize, **kwargs) -> plt.Figure
        - vals: The values used as inputs for each figure: shape (nrow, ncol)
        - col_labels: Optional. The labels for each column in the table.
        - row_labels: Optional. The labels for each row in the table.
        - figsize: Optional. The size of each figure in inches.
        - dpi: Optional. The resolution of each figure in dots per inch.
        - label_thickness: Optional. The thickness of the column and row labels, in inches.
        - h_spacing: Optional. The horizontal spacing between figures in the table.
        - v_spacing: Optional. The vertical spacing between figures in the table.
        - **kwargs: Optional. Additional keyword arguments passed to the plotting function.

    Returns:
        - fig_table: The table of figures as a single image.
    """
    if col_labels is not None:
        assert vals.shape[1] == len(col_labels)
    if row_labels is not None:
        assert vals.shape[0] == len(row_labels)
    
    ims = np.empty((vals.shape[0], vals.shape[1]), dtype=object)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            fig = plotting_function(vals[i,j], figsize=figsize, **kwargs)
            ims[i,j] = fig2img(fig, dpi=dpi)

    if col_labels is not None:
        col_labels = [fig2img(make_label_fig(text, figsize=(figsize[0], label_thickness)), dpi=dpi) for text in col_labels]
    if row_labels is not None:
        row_labels = [fig2img(make_label_fig(text, figsize=(label_thickness, figsize[1]), rotation=90), dpi=dpi) for text in row_labels]

    return fig_table_from_ims(ims, col_label_ims=col_labels, row_label_ims=row_labels, h_spacing=h_spacing, v_spacing=v_spacing)