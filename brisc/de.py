from __future__ import annotations
import numpy as np
import os
import polars as pl
import signal
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from .utils import bonferroni, check_bounds, check_dtype, check_type, \
    import_cython, fdr, plural, to_tuple_checked


class DE:
    """
    Differential expression results returned by `Pseudobulk.de()`.
    """

    def __init__(self,
                 source: str | Path | None = None,
                 /,
                 *,
                 table: pl.DataFrame | None = None,
                 voom_weights: dict[str, pl.DataFrame] | None = None,
                 voom_plot_data: dict[str, pl.DataFrame] | None = None) -> \
            None:
        """
        Initialize the DE object.

        Args:
            source: a directory containing a DE object saved with `save()`.
                    Mutually exclusive with `table`, `voom_weights`, and
                    `voom_plot_data`.
            table: a polars DataFrame containing the DE results, with columns:

                   - `cell_type`: the cell type in which DE was tested
                   - `coefficient`: the coefficient (or contrast) for which DE
                     was tested
                   - `gene`: the gene for which DE was tested
                   - `logFC`: the log2 fold change of the gene, i.e. its effect
                     size
                   - `SE`: the standard error of the effect size
                   - `LCI`: the lower 95% confidence interval of the effect
                     size
                   - `UCI`: the upper 95% confidence interval of the effect
                     size
                   - `AveExpr`: the gene's average expression in this cell
                     type, in log CPM
                   - `p`: the DE p-value
                   - `Bonferroni`: the Bonferroni-corrected DE p-value
                   - `FDR`: the FDR q-value for the DE

                   Mutually exclusive with `source`.
            voom_weights: an optional {cell_type: DataFrame} dictionary of voom
                          weights, where rows are genes and columns are
                          samples. The first column of each cell type's
                          DataFrame, 'gene', contains the gene names. Mutually
                          exclusive with `source`.
            voom_plot_data: an optional {cell_type: DataFrame} dictionary of
                            info necessary to construct a voom plot with
                            `plot_voom()`. Mutually exclusive with `source`.
        """
        if source is not None and table is not None:
            error_message = 'only one of source and table can be specified'
            raise ValueError(error_message)
        if source is not None:
            check_type(source, 'source', (str, Path),
                       'a string or pathlib.Path')
            if voom_plot_data is not None:
                error_message = (
                    'voom_plot_data cannot be specified when source is '
                    'specified')
                raise ValueError(error_message)
            if voom_weights is not None:
                error_message = (
                    'voom_weights cannot be specified when source is '
                    'specified')
                raise ValueError(error_message)
            source = str(source)
            if not os.path.isdir(source):
                if os.path.isfile(source):
                    error_message = \
                        f'{source!r} must be a directory, not a file'
                    raise NotADirectoryError(error_message)
                else:
                    error_message = \
                        f'DE results directory {source!r} does not exist'
                    raise FileNotFoundError(error_message)
            cell_types_file = f'{source}/cell_types.txt'
            if os.path.exists(cell_types_file):
                cell_types = [line.rstrip('\n') for line in
                              open(cell_types_file)]
                voom_weights = {cell_type: pl.read_parquet(
                    os.path.join(source, f'{cell_type.replace("/", "-")}.'
                                         f'voom_weights.parquet'))
                    for cell_type in cell_types}
                voom_plot_data = {cell_type: pl.read_parquet(
                    os.path.join(source, f'{cell_type.replace("/", "-")}.'
                                         f'voom_plot_data.parquet'))
                    for cell_type in cell_types}
            else:
                voom_weights = None
                voom_plot_data = None
            table = pl.read_parquet(os.path.join(source, 'table.parquet'))
        elif table is not None:
            check_type(table, 'table', pl.DataFrame, 'a polars DataFrame')
            if voom_weights is not None:
                if voom_plot_data is None:
                    error_message = (
                        'voom_plot_data must be specified when voom_weights '
                        'is specified')
                    raise ValueError(error_message)
                check_type(voom_weights, 'voom_weights', dict, 'a dictionary')
                if voom_weights.keys() != voom_plot_data.keys():
                    error_message = (
                        'voom_weights and voom_plot_data must have matching '
                        'cell types (keys)')
                    raise ValueError(error_message)
                for key in voom_weights:
                    if not isinstance(key, str):
                        error_message = (
                            f'all keys of voom_weights and voom_plot_data '
                            f'must be strings (cell types), but they contain '
                            f'a key of type {type(key).__name__!r}')
                        raise TypeError(error_message)
            if voom_plot_data is not None:
                if voom_weights is None:
                    error_message = (
                        'voom_weights must be specified when voom_plot_data '
                        'is specified')
                    raise ValueError(error_message)
                check_type(voom_plot_data, 'voom_plot_data', dict,
                           'a dictionary')
        else:
            error_message = 'either source or table must be specified'
            raise ValueError(error_message)
        self.table = table
        self.voom_weights = voom_weights
        self.voom_plot_data = voom_plot_data

    def __repr__(self) -> str:
        """
        Get a string representation of this DE object.

        Returns:
            A string summarizing the object.
        """
        num_cell_types = self.table['cell_type'].n_unique()
        descr = (
            f'DE object with {len(self.table):,} '
            f'{"entries" if len(self.table) != 1 else "entry"} across '
            f'{num_cell_types:,} {plural("cell type", num_cell_types)}:\n'
            f'{self.table}')
        return descr

    def __eq__(self, other: DE) -> bool:
        """
        Test for equality with another DE object.

        Args:
            other: the other DE object to test for equality with

        Returns:
            Whether the two DE objects are identical.
        """
        if not isinstance(other, DE):
            error_message = (
                f'the left-hand operand of `==` is a DE object, but '
                f'the right-hand operand has type {type(other).__name__!r}')
            raise TypeError(error_message)
        return self.table.equals(other.table) and \
            (other.voom_weights is None if self.voom_weights is None else
             other.voom_weights is not None and
             self.voom_weights.keys() == other.voom_weights.keys() and
             all(self.voom_weights[cell_type].equals(
                     other.voom_weights[cell_type]) and
                 self.voom_plot_data[cell_type].equals(
                     other.voom_plot_data[cell_type])
                 for cell_type in self.voom_weights))

    @property
    def groups(self) -> dict[str, tuple[str, ...] | None] | None:
        """
        The groups used by `voomByGroup` for each cell type: a dictionary
        mapping cell type names to group names used by voomByGroup for that
        cell type, or `None` if voomByGroup was not used for that cell type.
        If `Pseudobulk.DE()` was called with `return_voom_info=False`, `groups`
        will be `None` instead of a dictionary.
        """
        return {cell_type: None if 'xy_x' in data.columns else
                           tuple(column[5:] for column in data.columns
                                 if column[:4] == 'xy_x')
                    for cell_type, data in self.voom_plot_data.items()} \
            if self.voom_plot_data is not None else None

    def save(self, directory: str | Path, /, *, overwrite: bool = False) -> \
            None:
        """
        Save a DE object to `directory` (which must not exist unless
        `overwrite=True`, and will be created) with the table at
        `table.parquet`.

        If the DE object contains voom info (i.e. was created with
        `return_voom_info=True` in `Pseudobulk.de()`, the default), also saves
        each cell type's voom weights and voom plot data to
        f'{cell_type}_voom_weights.parquet' and
        f'{cell_type}_voom_plot_data.parquet', as well as a text file,
        cell_types.txt, containing the cell types.

        Args:
            directory: the directory to save the DE object to
            overwrite: if `False`, raises an error if the directory exists; if
                       `True`, overwrites files inside it as necessary
        """
        check_type(directory, 'directory', (str, Path),
                   'a string or pathlib.Path')
        directory = str(directory)
        check_type(overwrite, 'overwrite', bool, 'Boolean')
        if not overwrite and os.path.exists(directory):
            if os.path.isfile(directory):
                error_message = (
                    f'cannot save to the directory {directory!r} because it '
                    f'already exists as a file')
                raise FileExistsError(error_message)
            else:
                error_message = (
                    f'directory {directory!r} already exists; set '
                    f'overwrite=True to overwrite')
                raise FileExistsError(error_message)
        os.makedirs(directory, exist_ok=overwrite)
        self.table.write_parquet(os.path.join(directory, 'table.parquet'))
        if self.voom_weights is not None:
            with open(os.path.join(directory, 'cell_types.txt'), 'w') as f:
                print('\n'.join(self.voom_weights), file=f)
            for cell_type in self.voom_weights:
                escaped_cell_type = cell_type.replace('/', '-')
                self.voom_weights[cell_type].write_parquet(
                    os.path.join(directory, f'{escaped_cell_type}.'
                                            f'voom_weights.parquet'))
                self.voom_plot_data[cell_type].write_parquet(
                    os.path.join(directory, f'{escaped_cell_type}.'
                                            f'voom_plot_data.parquet'))

    def get_hits(self,
                 *,
                 significance_column: str = 'FDR',
                 threshold: int | float | np.integer | np.floating = 0.05,
                 num_top_hits: int | np.integer | None = None) -> pl.DataFrame:
        """
        Get all (or the top) differentially expressed genes.

        Args:
            significance_column: the name of a numeric column of `self.table`
                                 to determine significance from
            threshold: the significance threshold corresponding to
                       `significance_column`
            num_top_hits: the number of top hits to report for each cell type;
                          if `None`, report all hits

        Returns:
            The `table` attribute of this DE object, subset to (top) DE hits.
        """
        check_type(significance_column, 'significance_column', str, 'a string')
        if significance_column not in self.table:
            error_message = (
                f'significance_column ({significance_column!r}) is not a '
                f'column of self.table')
            raise ValueError(error_message)
        check_dtype(self.table[significance_column],
                    f'self.table[{significance_column!r}]', 'floating-point')
        check_type(threshold, 'threshold', (int, float),
                   'a number > 0 and ≤ 1')
        check_bounds(threshold, 'threshold', 0, 1, left_open=True)
        if num_top_hits is not None:
            check_type(num_top_hits, 'num_top_hits', int, 'a positive integer')
            check_bounds(num_top_hits, 'num_top_hits', 1)
        hits = self.table.filter(pl.col(significance_column) < threshold)
        if num_top_hits is not None:
            hits = hits\
                .group_by('cell_type', maintain_order=True)\
                .agg(pl.all().bottom_k_by(significance_column, num_top_hits))\
                .explode(pl.exclude('cell_type'))
        return hits.sort(significance_column)

    def get_num_hits(self,
                     *,
                     significance_column: str = 'FDR',
                     threshold: int | float | np.integer |
                                np.floating = 0.05) -> pl.DataFrame:
        """
        Get the number of differentially expressed genes in each cell type.

        Args:
            significance_column: the name of a numeric column of `self.table`
                                 to determine significance from
            threshold: the significance threshold corresponding to
                       `significance_column`

        Returns:
            A DataFrame with one row per cell type and two columns:
            'cell_type' and 'num_hits'.
        """
        check_type(significance_column, 'significance_column', str, 'a string')
        if significance_column not in self.table:
            error_message = (
                f'significance_column ({significance_column!r}) is not a '
                f'column of self.table')
            raise ValueError(error_message)
        check_dtype(self.table[significance_column],
                    f'self.table[{significance_column!r}]', 'floating-point')
        check_type(threshold, 'threshold', (int, float),
                   'a number > 0 and ≤ 1')
        check_bounds(threshold, 'threshold', 0, 1, left_open=True)
        return self.table\
            .lazy()\
            .filter(pl.col(significance_column) < threshold)\
            .group_by('cell_type')\
            .agg(num_hits=pl.len())\
            .sort('cell_type')\
            .collect()

    def plot_voom(self,
                  cell_type: str,
                  filename: str | Path | None = None,
                  /,
                  *,
                  ax: 'Axes' | None = None,
                  figure_kwargs: dict[str, Any] | None = None,
                  point_color: Color | dict[str, Color] | None = None,
                  point_size: int | float | np.integer | np.floating |
                              dict[str, int | float | np.integer |
                                        np.floating] = 1,
                  line_color: Color | dict[str, Color] | None = None,
                  line_width: int | float | np.integer | np.floating |
                              dict[str, int | float | np.integer |
                                        np.floating] = 1.5,
                  scatter_kwargs: dict[str, Any] | None |
                                  dict[str, dict[str, Any] | None] = None,
                  plot_kwargs: dict[str, Any] | None |
                               dict[str, dict[str, Any] | None] = None,
                  legend: bool = True,
                  legend_kwargs: dict[str, Any] | None = None,
                  title: str | None = None,
                  title_kwargs: dict[str, Any] | None = None,
                  xlabel: str | None = 'Average log2(count + 0.5)',
                  xlabel_kwargs: dict[str, Any] | None = None,
                  ylabel: str | None = 'sqrt(standard deviation)',
                  ylabel_kwargs: dict[str, Any] | None = None,
                  despine: bool = True,
                  savefig_kwargs: dict[str, Any] | None = None) -> None:
        """
        Generate a voom plot for a cell type that differential expression was
        calculated for.

        Voom plots consist of a scatter plot with one point per gene. They
        visualize how the mean expression of each gene across samples (x)
        relates to the gene's variation in expression across samples (y). The
        plot also includes a LOESS (also called LOWESS) curve, a type of
        non-linear curve fit, of the mean-variance (x-y) trend.

        Specifically, the x position of a gene's point is the average, across
        samples, of the base-2 logarithm of the gene's count in each sample,
        plus a pseudocount of 0.5: in other words, mean(log2(count + 0.5)).
        The y position is the square root of the standard deviation, across
        samples, of the gene's log counts per million after regressing out,
        across samples, the differential expression design matrix.

        When running differential expression with voomByGroup, voom is run
        separately within each group, so the voom plot will show a separate
        LOESS trendline for each group, with the points and trendlines for each
        group shown in distinct colors.

        Many arguments to this function can be either a single value or a
        dictionary mapping group names to values. Dictionaries can only be
        used with voomByGroup, and must contain every group. The group names
        can be viewed with `self.groups[cell_type]`.

        Args:
            cell_type: the cell type to generate the voom plot for
            filename: the file to save to. If `None`, generate the plot but do
                      not save it, which allows it to be shown interactively or
                      modified further before saving. Mutually exclusive with
                      `ax`.
            ax: the Matplotlib axes to plot onto; if `None`, create a new
                figure with Matplotlib's constrained layout and plot onto it.
                Mutually exclusive with `filename`.
            figure_kwargs: a dictionary of keyword arguments to be passed to
                           [`plt.figure()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html)
                           when `ax` is `None`, such as:

                           - `figsize`: a two-element sequence of the width and
                             height of the figure in inches. Defaults to
                             `[6.4, 4.8]`.
                           - `layout`: the layout mechanism used by Matplotlib
                             to avoid overlapping plot elements. Defaults to
                             `'constrained'`, instead of Matplotlib's default
                             of `None`.

            point_color: the color of the points in the voom plot. Can be a
                         single color or a dictionary mapping each of the group
                         names in `self.groups[cell_type]` to colors. When not
                         using voomByGroup, defaults to `'#666666'` (gray).
                         When using voomByGroup with two groups, defaults to
                         `'#666666'` for the first group in
                         `self.groups[cell_type]` and `'#FF6666'` (red) for the
                         second. When using voomByGroup with more than two
                         groups, must be specified manually. Can be any valid
                         Matplotlib color, like a hex string (e.g.
                         `'#FF0000'`), a named color (e.g. 'red'), a 3- or
                         4-element RGB/RGBA tuple of integers 0-255 or floats
                         0-1, or a single float 0-1 for grayscale.
            point_size: the size of the points in the voom plot. Can be a
                        single number or a dictionary mapping each of the group
                        names in `self.groups[cell_type]` to numbers.
            line_color: the color of the LOESS trendline. Can be a single color
                        or a dictionary mapping each of the group names in
                        `self.groups[cell_type]` to colors. When not using
                        voomByGroup, defaults to `'#000000'` (black). When
                        using voomByGroup with two groups, defaults to
                        `'#000000'` for the first group and `'#FF0000'` (red)
                        for the second. When using voomByGroup with more than
                        two groups, must be specified manually. Can be any
                        valid Matplotlib color, like a hex string (e.g.
                        `'#FF0000'`), a named color (e.g. 'red'), a 3- or
                        4-element RGB/RGBA tuple of integers 0-255 or floats
                        0-1, or a single float 0-1 for grayscale.
            line_width: the width of the LOESS trendline. Can be a single
                        number or a dictionary mapping each of the group names
                        in `self.groups[cell_type]` to numbers.
            scatter_kwargs: a dictionary (or dictionary mapping each of the
                            group names in `self.groups[cell_type]` to
                            dictionaries) of keyword arguments to be passed to
                            [`ax.scatter()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html),
                            such as:

                            - `rasterized`: whether to convert the scatter plot
                              points to a raster (bitmap) image when saving to
                              a vector format like PDF. Defaults to `True`,
                              instead of Matplotlib's default of `False`.
                            - `marker`: the shape to use for plotting each gene
                            - `alpha`: the transparency of each point
                            - `linewidths` and `edgecolors`: the width and
                              color of the borders around each marker. These
                              are absent by default (`linewidths=0`,
                              `edgecolors=(0, 0, 0, 0)`), unlike Matplotlib's
                              default. Both arguments can be either single
                              values or sequences.
                            - `zorder`: the order in which the genes are
                              plotted, with higher values appearing on top of
                              lower ones.

                            Specifying `s` or `c`/`color`/`norm`/`vmin`/`vmax`
                            will raise an error, since these arguments conflict
                            with the `point_size` and `point_color` arguments,
                            respectively.
            plot_kwargs: a dictionary (or dictionary mapping each of the group
                         names in `self.groups[cell_type]` to dictionaries) of
                         keyword arguments to be passed to
                         [`ax.plot()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html)
                         when plotting the trendlines, such as `linestyle='--'`
                         for dashed trendlines. Specifying `color`/`c` or
                         `linewidth` will raise an error, since these arguments
                         conflict with the `line_color` and `line_width`
                         arguments, respectively.
            legend: whether to add a legend with the colors for each group when
                    using voomByGroup. Only `legend=False` has an effect, and
                    it can only be specified when using voomByGroup. Without
                    groups, there will never be a legend, so specifying
                    `legend=False` would be redundant.
            legend_kwargs: a dictionary of keyword arguments to be passed to
                           [`ax.legend()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html)
                           to modify the legend, such as:

                           - `loc`, `bbox_to_anchor`, and `bbox_transform` to
                             set its location.
                           - `prop`, `fontsize`, and `labelcolor` to set its
                             font properties
                           - `facecolor` and `framealpha` to set its background
                             color and transparency
                           - `frameon=True` or `edgecolor` to add or color
                             its border. `frameon` defaults to `False`, instead
                             of Matplotlib's default of `True`.
                           - `title` to add a legend title

                           Can only be specified when using voomByGroup with
                           `legend=True`.
            title: the title of the plot, or `None` to not add a title
            title_kwargs: a dictionary of keyword arguments to be passed to
                          [`ax.set_title()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_title.html)
                          to control text properties, such as `color` and
                          `size`. Can only be specified when `title` is not
                          `None`.
            xlabel: the x-axis label, or `None` to not label the x-axis
            xlabel_kwargs: a dictionary of keyword arguments to be passed to
                           [`ax.set_xlabel()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xlabel.html)
                           to control text properties, such as `color` and
                           `size`. Can only be specified when `xlabel` is not
                           `None`.
            ylabel: the y-axis label, or `None` to not label the y-axis
            ylabel_kwargs: a dictionary of keyword arguments to be passed to
                           [`ax.set_ylabel()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_ylabel.html)
                           to control text properties, such as `color` and
                           `size`. Can only be specified when `ylabel` is not
                           `None`.
            despine: whether to remove the top and right spines (borders of the
                     plot area) from the voom plot
            savefig_kwargs: a dictionary of keyword arguments to be passed to
                            [`plt.savefig()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html),
                            such as:

                            - `dpi`: defaults to 300 instead of Matplotlib's
                              default of 150
                            - `bbox_inches`: the bounding box of the portion of
                              the figure to save; defaults to 'tight' (crop out
                              any blank borders) instead of Matplotlib's
                              default of `None` (save the entire figure)
                            - `pad_inches`: the number of inches of padding to
                              add on each of the four sides of the figure when
                              saving. Defaults to 'layout' (use the padding
                              from the constrained layout engine), instead of
                              Matplotlib's default of 0.1.
                            - `transparent`: whether to save with a transparent
                              background; defaults to `True` if saving to a PDF
                              (i.e. when `filename` ends with `'.pdf'`) and
                              `False` otherwise, instead of Matplotlib's
                              default of always being `False`.

                            Can only be specified when `filename` is specified.
        """
        # Import matplotlib
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            import matplotlib.pyplot as plt
        finally:
            signal.signal(signal.SIGINT, original_sigint_handler)

        # Check that this DE object contains `voom_plot_data`, the data
        # necessary to generate the voom plot from (scatter-plot points and
        # LOESS trendlines)
        if self.voom_plot_data is None:
            error_message = (
                'this DE object does not contain the voom_plot_data '
                'attribute, which is necessary to generate voom plots; re-run '
                'Pseudobulk.de() with return_voom_info=True to include this '
                'attribute')
            raise AttributeError(error_message)

        # Check that `cell_type` is a cell type in this DE object
        check_type(cell_type, 'cell_type', str, 'a string')
        if cell_type not in self.voom_plot_data:
            error_message = \
                f'cell_type {cell_type!r} is not a cell type in this DE object'
            raise ValueError(error_message)

        # Get the voom plot data for this cell type
        voom_plot_data = self.voom_plot_data[cell_type]

        # Get the voomByGroup groups for this cell type (`None` if voomByGroup
        # was not used)
        groups = None if 'xy_x' in voom_plot_data.columns else \
            tuple(column[5:] for column in voom_plot_data.columns
                  if column[:4] == 'xy_x')

        # If `filename` was specified, check that `ax` is `None`, that
        # `filename` is a string or `pathlib.Path`, and that its base directory
        # exists; if `filename` is `None`, make sure `savefig_kwargs` is also
        # `None`
        if filename is not None:
            if ax is not None:
                error_message = (
                    'filename and ax are mutually exclusive; if you specified '
                    'ax in order to create multiple subplots, make a separate '
                    'savefig() call once all subplots are created')
                raise ValueError(error_message)
            check_type(filename, 'filename', (str, Path),
                       'a string or pathlib.Path')
            directory = os.path.dirname(filename)
            if directory and not os.path.isdir(directory):
                error_message = (
                    f'{filename} refers to a file in the directory '
                    f'{directory!r}, but this directory does not exist')
                raise NotADirectoryError(error_message)
            filename = str(filename)
        elif savefig_kwargs is not None:
            error_message = 'savefig_kwargs must be None when filename is None'
            raise ValueError(error_message)

        # If `figure_kwargs` was specified, check that `ax` is `None`
        if figure_kwargs is not None and ax is not None:
            error_message = (
                'figure_kwargs must be None when ax is not None, since a new '
                'figure does not need to be generated when plotting onto an '
                'existing axis')
            raise ValueError(error_message)

        # Check that `point_color`, `point_size`, `line_color`, and
        # `line_width` are only dictionaries when using voomByGroup, and that
        # their keys are exactly the group names (in any order)
        for arg, arg_name in ((point_color, 'point_color'),
                              (point_size, 'point_size'),
                              (line_color, 'line_color'),
                              (line_width, 'line_width')):
            if isinstance(arg, dict):
                if groups is None:
                    error_message = (
                        f'{arg_name} can only be a dictionary when using '
                        f'voomByGroup')
                    raise ValueError(error_message)
                if set(arg) != set(groups):
                    error_message = (
                        f'{arg_name} is a dictionary, but its keys differ '
                        f'from the groups in self.groups[{cell_type!r}]: '
                        f'{", ".join(map(repr, groups))}')
                    raise ValueError(error_message)

        # Check that `point_color` and `line_color` are valid Matplotlib colors
        # or dictionaries thereof, and convert them to hex. Or, if `None`, set
        # to their default value if there are no groups or exactly two groups.
        point_color_is_dict = isinstance(point_color, dict)
        if point_color is None:
            if groups is None:
                point_color = '#666666'
            elif len(groups) == 2:
                point_color = {groups[0]: '#666666', groups[1]: '#FF6666'}
                point_color_is_dict = True
            else:
                error_message = (
                    f'point_color must be specified manually when there are '
                    f'three or more groups; here, there are {len(groups)!r}')
                raise ValueError(error_message)
        elif point_color_is_dict:
            for group, group_point_color in point_color.items():
                if not plt.matplotlib.colors.is_color_like(group_point_color):
                    error_message = (
                        f'point_color[{group!r}] is not a valid Matplotlib '
                        f'color or sequence of valid colors')
                    raise ValueError(error_message)
            point_color = {
                group: plt.matplotlib.colors.to_hex(group_point_color)
                for group, group_point_color in point_color.items()}
        else:
            if not plt.matplotlib.colors.is_color_like(point_color):
                error_message = (
                    f'point_color is not a valid Matplotlib color or '
                    f'sequence of valid colors')
                raise ValueError(error_message)
            point_color = plt.matplotlib.colors.to_hex(point_color)
        line_color_is_dict = isinstance(line_color, dict)
        if line_color is None:
            if groups is None:
                line_color = '#000000'
            elif len(groups) == 2:
                line_color = {groups[0]: '#000000', groups[1]: '#FF0000'}
                line_color_is_dict = True
            else:
                error_message = (
                    f'line_color must be specified manually when there are '
                    f'three or more groups; here, there are {len(groups)!r}')
                raise ValueError(error_message)
        elif line_color_is_dict:
            for group, group_line_color in line_color.items():
                if not plt.matplotlib.colors.is_color_like(group_line_color):
                    error_message = (
                        f'line_color[{group!r}] is not a valid Matplotlib '
                        f'color or sequence of valid colors')
                    raise ValueError(error_message)
            line_color = {
                group: plt.matplotlib.colors.to_hex(group_line_color)
                for group, group_line_color in line_color.items()}
        else:
            if not plt.matplotlib.colors.is_color_like(line_color):
                error_message = (
                    f'line_color is not a valid Matplotlib color or '
                    f'sequence of valid colors')
                raise ValueError(error_message)
            line_color = plt.matplotlib.colors.to_hex(line_color)

        # Check that `point_size` and `line_width` are positive numbers or
        # dicts thereof
        point_size_is_dict = isinstance(point_size, dict)
        line_width_is_dict = isinstance(line_width, dict)
        for number, number_name, is_dict in (
                (point_size, 'point_size', point_size_is_dict),
                (line_width, 'line_width', line_width_is_dict)):
            if is_dict:
                for group, group_number in number.items():
                    check_type(group_number, f'{number_name}[{group!r}]',
                               (int, float), 'a positive number')
                    check_bounds(group_number, f'{number_name}[{group!r}]', 0,
                                 left_open=True)
            else:
                check_type(number, number_name, (int, float),
                           'a positive number')
                check_bounds(number, number_name, 0, left_open=True)

        # For each of the kwargs arguments, if the argument was specified,
        # check that it is a dictionary and that all its keys are strings.
        for kwargs, kwargs_name in ((figure_kwargs, 'figure_kwargs'),
                                    (scatter_kwargs, 'scatter_kwargs'),
                                    (plot_kwargs, 'plot_kwargs'),
                                    (legend_kwargs, 'legend_kwargs'),
                                    (xlabel_kwargs, 'xlabel_kwargs'),
                                    (ylabel_kwargs, 'ylabel_kwargs'),
                                    (title_kwargs, 'title_kwargs'),
                                    (savefig_kwargs, 'savefig_kwargs')):
            if kwargs is not None:
                check_type(kwargs, kwargs_name, dict, 'a dictionary')
                for key in kwargs:
                    if not isinstance(key, str):
                        error_message = (
                            f'all keys of {kwargs_name} must be strings, but '
                            f'it contains a key of type '
                            f'{type(key).__name__!r}')
                        raise TypeError(error_message)

        # If using voomByGroup, for each of `scatter_kwargs` and `plot_kwargs`,
        # if the kwarg was specified, check that either all keys are group
        # names and in the correct order, or that no keys are group names. If
        # all keys are group names, check that all values are either `None` or
        # dictionaries with all-string keys, and make note that the kwargs is a
        # nested dict.
        scatter_kwargs_is_nested_dict = False
        plot_kwargs_is_nested_dict = False
        if groups is not None:
            for kwargs, kwargs_name in ((scatter_kwargs, 'scatter_kwargs'),
                                        (plot_kwargs, 'plot_kwargs')):
                if kwargs is not None:
                    if tuple(kwargs) == groups:
                        # The kwargs's keys exactly match the group names
                        for key, value in kwargs.items():
                            if value is not None:
                                check_type(value, f'{kwargs_name}[{key!r}]',
                                           dict, 'a dictionary')
                                for inner_key in value:
                                    if not isinstance(inner_key, str):
                                        error_message = (
                                            f'all keys of '
                                            f'{kwargs_name}[{key!r}] must be '
                                            f'strings, but it contains a key '
                                            f'of type '
                                            f'{type(inner_key).__name__!r}')
                                        raise TypeError(error_message)
                        if kwargs is scatter_kwargs:
                            scatter_kwargs_is_nested_dict = True
                        else:
                            plot_kwargs_is_nested_dict = True
                    else:
                        # Check that none of the kwargs's keys are group names
                        for group in groups:
                            if group in kwargs:
                                if set(groups) == set(kwargs):
                                    error_message = (
                                        f'{kwargs_name}.keys() does have the '
                                        f'same groups as '
                                        f'self.groups[{cell_type!r}], but '
                                        f'they are in a different order')
                                    raise ValueError(error_message)
                                else:
                                    error_message = (
                                        f'some keys of {kwargs_name}.keys() '
                                        f'are groups in '
                                        f'self.groups[{cell_type!r}], but '
                                        f'others are not')
                                    raise ValueError(error_message)

        # Override the defaults for certain keys of `scatter_kwargs`
        default_scatter_kwargs = dict(rasterized=True, linewidths=0,
                                      edgecolors=(0, 0, 0, 0))
        if scatter_kwargs_is_nested_dict:
            scatter_kwargs = {
                group: default_scatter_kwargs | group_scatter_kwargs
                       if group_scatter_kwargs is not None else
                       default_scatter_kwargs
                for group, group_scatter_kwargs in scatter_kwargs.items()}
        else:
            scatter_kwargs = default_scatter_kwargs | scatter_kwargs \
                if scatter_kwargs is not None else default_scatter_kwargs

        # Set `plot_kwargs` to `{}` if it is `None`, or set the `None` values
        # of `plot_kwargs` to `{}` if `plot_kwargs` is a nested dict
        if plot_kwargs is None:
            plot_kwargs = {}
        elif plot_kwargs_is_nested_dict:
            plot_kwargs = {
                group: group_plot_kwargs
                       if group_plot_kwargs is not None else {}
                for group, group_plot_kwargs in plot_kwargs.items()}

        # Check that `scatter_kwargs` does not contain the `s` or
        # `c`/`color`/`norm`/`vmin`/`vmax` keys and that `plot_kwargs` does
        # not contain the `c`/`color`/`norm`/`vmin`/`vmax` or `linewidth` keys,
        # or that their values do not contain these keys if a nested dict
        for kwargs, kwargs_name, alternate_color, is_nested_dict in (
                (scatter_kwargs, 'scatter_kwargs', 'point_color',
                 scatter_kwargs_is_nested_dict),
                (plot_kwargs, 'plot_kwargs', 'line_color',
                 plot_kwargs_is_nested_dict)):
            bad_keys = (('linewidth', 'line_width')
                        if kwargs is plot_kwargs else ('s', 'point_size'),
                        ('c', alternate_color),
                        ('color', alternate_color),
                        ('norm', alternate_color),
                        ('vmin', alternate_color),
                        ('vmax', alternate_color))
            if is_nested_dict:
                for key, value in kwargs.items():
                    for bad_key, alternate_argument in bad_keys:
                        if bad_key in value:
                            error_message = (
                                f'{bad_key!r} cannot be specified as a key in '
                                f'{kwargs_name}[{key!r}]; specify the '
                                f'{alternate_argument} argument instead')
                            raise ValueError(error_message)
            else:
                for bad_key, alternate_argument in bad_keys:
                    if bad_key in kwargs:
                        error_message = (
                            f'{bad_key!r} cannot be specified as a key in '
                            f'{kwargs_name}; specify the {alternate_argument} '
                            f'argument instead')
                        raise ValueError(error_message)

        # Check that `legend` is Boolean. If not using voomByGroup, check that
        # the user did not specify `legend=False`.
        check_type(legend, 'legend', bool, 'Boolean')
        if groups is None:
            if not legend:
                error_message = (
                    'legend=False cannot be specified when there are no '
                    'groups, since it would be redundant: without groups, '
                    'there will never be a legend')
                raise ValueError(error_message)

        # Override the defaults for certain values of `legend_kwargs`; check
        # that it is `None` when not using a legend
        default_legend_kwargs = dict(frameon=False)
        if legend_kwargs is not None:
            if groups is None:
                error_message = (
                    'legend_kwargs cannot be specified when there are no '
                    'groups, since there will not be a legend')
                raise ValueError(error_message)
            if not legend:
                error_message = \
                    'legend_kwargs cannot be specified when legend=False'
                raise ValueError(error_message)
            legend_kwargs = default_legend_kwargs | legend_kwargs
        else:
            legend_kwargs = default_legend_kwargs

        # If `title` was specified, check that it is a string
        if title is not None:
            check_type(title, 'title', str, 'a string')

        # Check that `title_kwargs` is `None` when `title` is `None`
        if title is None and title_kwargs is not None:
            error_message = 'title_kwargs cannot be specified when title=None'
            raise ValueError(error_message)

        # Check that `xlabel` is a string or `None`; if `None`, check that
        # `xlabel_kwargs` is `None` as well. Ditto for `ylabel`.
        for arg, arg_name, arg_kwargs in (
                (xlabel, 'xlabel', xlabel_kwargs),
                (ylabel, 'ylabel', ylabel_kwargs)):
            if arg is not None:
                check_type(arg, arg_name, str, 'a string')
            elif arg_kwargs is not None:
                error_message = \
                    f'{arg_name}_kwargs must be None when {arg_name} is None'
                raise ValueError(error_message)

        # Check that `despine` is Boolean
        check_type(despine, 'despine', bool, 'Boolean')

        # If `ax` is `None`, create a new figure with
        # `constrained_layout=True`; otherwise, check that it is a Matplotlib
        # axis
        fig = None
        try:
            if ax is None:
                default_figure_kwargs = dict(layout='constrained')
                figure_kwargs = default_figure_kwargs | figure_kwargs \
                    if figure_kwargs is not None else default_figure_kwargs
                fig = plt.figure(**figure_kwargs)
                ax = fig.gca()
            else:
                check_type(ax, 'ax', plt.Axes, 'a Matplotlib axis')
            if groups is not None:
                if legend:
                    legend_patches = []
                for group in groups:
                    # Get this group's point size, point color, line color,
                    # line width, plot kwargs, and scatter kwargs
                    group_point_size = point_size[group] \
                        if point_size_is_dict else point_size
                    group_point_color = point_color[group] \
                        if point_color_is_dict else point_color
                    group_line_color = line_color[group] \
                        if line_color_is_dict else line_color
                    group_line_width = line_width[group] \
                        if line_width_is_dict else line_width
                    group_scatter_kwargs = scatter_kwargs[group] \
                        if scatter_kwargs_is_nested_dict else scatter_kwargs
                    group_plot_kwargs = plot_kwargs[group] \
                        if plot_kwargs_is_nested_dict else plot_kwargs

                    # Plot the scatter plot for this group
                    ax.scatter(voom_plot_data[f'xy_x_{group}'].drop_nulls(),
                               voom_plot_data[f'xy_y_{group}'].drop_nulls(),
                               s=group_point_size, c=group_point_color,
                               **group_scatter_kwargs)

                    # Plot the LOESS trendline for this group
                    ax.plot(voom_plot_data[f'line_x_{group}'].drop_nulls(),
                            voom_plot_data[f'line_y_{group}'].drop_nulls(),
                            c=group_line_color, linewidth=group_line_width,
                            **group_plot_kwargs)

                    # Create a rectangle for the legend for this group, where
                    # the border matches the color of the trendline and the
                    # fill matches the color of the scatter plot points
                    if legend:
                        legend_patches.append(plt.matplotlib.patches.Patch(
                            facecolor=group_point_color,
                            edgecolor=group_line_color,
                            linewidth=group_line_width, label=group))

                # Add the legend
                if legend:
                    ax.legend(handles=legend_patches, **legend_kwargs)
            else:
                # Plot the scatter plot
                ax.scatter(voom_plot_data['xy_x'], voom_plot_data['xy_y'],
                           s=point_size, c=point_color, **scatter_kwargs)

                # Plot the LOESS trendline
                ax.plot(voom_plot_data['line_x'], voom_plot_data['line_y'],
                        c=line_color, linewidth=line_width, **plot_kwargs)

            # Add the title and axis labels
            if xlabel is not None:
                if xlabel_kwargs is None:
                    xlabel_kwargs = {}
                ax.set_xlabel(xlabel, **xlabel_kwargs)
            if ylabel is not None:
                if ylabel_kwargs is None:
                    ylabel_kwargs = {}
                ax.set_ylabel(ylabel, **ylabel_kwargs)
            if title is not None:
                if title_kwargs is None:
                    title_kwargs = {}
                ax.set_title(title, **title_kwargs)

            # Despine, if specified
            if despine:
                spines = ax.spines
                spines['top'].set_visible(False)
                spines['right'].set_visible(False)

            # Save; override the defaults for certain keys of `savefig_kwargs`
            if filename is not None:
                default_savefig_kwargs = \
                    dict(dpi=300, bbox_inches='tight', pad_inches='layout',
                         transparent=filename.endswith('.pdf'))
                savefig_kwargs = default_savefig_kwargs | savefig_kwargs \
                    if savefig_kwargs is not None else default_savefig_kwargs
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    fig.savefig(filename, **savefig_kwargs)
                plt.close(fig)
        except:
            # If we made a new figure, make sure to close it if there's an
            # exception (but not if there was no error and `filename` is
            # `None`, in case the user wants to modify it further before
            # saving)
            if fig is not None:
                plt.close(fig)
            raise

    def plot_volcano(self,
                     cell_type: str,
                     filename: str | Path | None = None,
                     /,
                     *,
                     ax: 'Axes' | None = None,
                     figure_kwargs: dict[str, Any] | None = None,
                     x_column: str = 'logFC',
                     y_column: str = 'p',
                     significance_column: str = 'FDR',
                     threshold: int | float | np.integer | np.floating = 0.05,
                     genes_to_label: int | np.integer | str | Iterable[str] |
                                     None = 10,
                     attraction: float = 5e-2,
                     repulsion: float = 1e-6,
                     max_iter: int = 4000,
                     seed: int = 0,
                     padding: float = 4.0,
                     box_padding: float = 4.0,
                     fontsize: float = 10,
                     linewidth: float = 0.6,
                     linecolor: Color = '0.45',
                     upregulated_size: int | float | np.integer |
                                       np.floating = 6,
                     downregulated_size: int | float | np.integer |
                                         np.floating = 6,
                     non_significant_size: int | float | np.integer |
                                           np.floating = 4,
                     upregulated_color: Color = '#FC4E07',
                     downregulated_color: Color = '#00AFBB',
                     non_significant_color: Color = 'lightgray',
                     upregulated_scatter_kwargs: dict[str, Any] | None = None,
                     downregulated_scatter_kwargs: dict[str, Any] |
                                                   None = None,
                     non_significant_scatter_kwargs: dict[str, Any] |
                                                     None = None,
                     legend: bool = True,
                     legend_kwargs: dict[str, Any] | None = None,
                     title: str | None = None,
                     title_kwargs: dict[str, Any] | None = None,
                     xlabel: str | None = '$log_2(FC)$',
                     xlabel_kwargs: dict[str, Any] | None = None,
                     ylabel: str | None = ...,
                     ylabel_kwargs: dict[str, Any] | None = None,
                     despine: bool = True,
                     savefig_kwargs: dict[str, Any] | None = None) -> None:
        """
        Generate a volcano plot of DE hits.

        Plots negative log p-values (or another `y_column`) on the y-axis
        against log fold changes (or another `x_column`) on the x-axis.
        Upregulated, downregulated and non-significant genes are plotted in
        three different colors based on `significance_column`.

        Args:
            cell_type: the cell type to generate the volcano plot for
            filename: the file to save to. If `None`, generate the plot but do
                      not save it, which allows it to be shown interactively or
                      modified further before saving. Mutually exclusive with
                      `ax`.
            ax: the Matplotlib axes to plot onto; if `None`, create a new
                figure with Matplotlib's constrained layout and plot onto it.
                Mutually exclusive with `filename`.
            figure_kwargs: a dictionary of keyword arguments to be passed to
                           [`plt.figure()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html)
                           when `ax` is `None`, such as:

                           - `figsize`: a two-element sequence of the width and
                             height of the figure in inches. Defaults to
                             Matplotlib's default of `[6.4, 4.8]`.
                           - `layout`: the layout mechanism used by Matplotlib
                             to avoid overlapping plot elements. Defaults to
                             `'constrained'`, instead of Matplotlib's default
                             of `None`.

            x_column: the name of a numeric column of `self.table` to plot on
                      the x-axis
            y_column: the name of a numeric column of `self.table` to plot the
                      negative log10 of on the y-axis
            significance_column: the name of a numeric column of `self.table`
                                 to determine significance from
            threshold: the significance threshold corresponding to
                       `significance_column`
            genes_to_label: a non-negative integer number of top DE genes
                            (according to `y_column`) to label, a name or
                            sequence of names of genes to label, or `None` (or
                            0) to not add labels. If an integer, only
                            significant DE genes (according to
                            `significance_column`) will be labeled, even if
                            `genes_to_label` is larger than the number of DE
                            genes.
            attraction: the strength of the spring that pulls each gene label
                        back toward its own point when it touches nothing;
                        larger values keep labels nearer their points.
            repulsion: the strength of the repulsion that pushes gene labels
                       away from one another and from points; larger values
                       spread the labels wider before they settle. Also the
                       standard deviation of the small initial jitter, so it
                       interacts with `seed`.
            max_iter: the maximum number of label force-simulation iterations
                      to run. The simulation also stops early once no label box
                      overlaps anything, so this is only reached for crowded
                      plots.
            seed: the seed for the small initial jitter that breaks ties
                  between symmetric labels. Fixed, so a given plot always
                  yields the same label layout.
            padding: the clearance kept, in pixels, between a gene label box
                     and any point (its own, another label's, or a background
                     gene). It is also the shortest connecting line drawn, so
                     no label sits on its own point without a visible line.
            box_padding: the padding, in pixels, added around each gene label's
                         text box. It sets the minimum gap kept between any two
                         labels.
            fontsize: the gene label font size, in points.
            linewidth: the width of the lines connecting each gene label to its
                       point, in points.
            linecolor: the color of the connecting lines. Can be any valid
                       Matplotlib color, like a hex string (e.g. `'#FF0000'`),
                       a named color (e.g. 'red'), a 3- or 4-element RGB/RGBA
                       tuple of integers 0-255 or floats 0-1, or a single float
                       0-1 for grayscale.
            upregulated_size: the size of each upregulated gene's point
            downregulated_size: the size of each downregulated gene's point
            non_significant_size: the size of each non-significant gene's point
            upregulated_color: the color of each upregulated gene's point. Can
                               be any valid Matplotlib color, like a hex string
                               (e.g. `'#FF0000'`), a named color (e.g. 'red'),
                               a 3- or 4-element RGB/RGBA tuple of integers
                               0-255 or floats 0-1, or a single float 0-1 for
                               grayscale.
            downregulated_color: the color of each downregulated gene's point.
                                 Can be any valid Matplotlib color, like a hex
                                 string (e.g. `'#FF0000'`), a named color (e.g.
                                 'red'), a 3- or 4-element RGB/RGBA tuple of
                                 integers 0-255 or floats 0-1, or a single
                                 float 0-1 for grayscale.
            non_significant_color: the color of each non-significant gene's
                                   point. Can be any valid Matplotlib color,
                                   like a hex string (e.g. `'#FF0000'`), a
                                   named color (e.g. 'red'), a 3- or 4-element
                                   RGB/RGBA tuple of integers 0-255 or floats
                                   0-1, or a single float 0-1 for grayscale.
            upregulated_scatter_kwargs: a dictionary of keyword arguments to be
                                        passed to
                                        [`ax.scatter()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html)
                                        for upregulated genes, such as:

                                        - `rasterized`: whether to convert the
                                          scatter plot points to a raster
                                          (bitmap) image when saving to a
                                          vector format like PDF. Defaults to
                                          `True`, instead of Matplotlib's
                                          default of `False`.
                                        - `marker`: the shape to use for
                                          plotting each gene
                                        - `alpha`: the transparency of each
                                          point
                                        - `linewidths` and `edgecolors`: the
                                          width and color of the borders around
                                          each marker. These are absent by
                                          default (`linewidths=0`,
                                          `edgecolors=(0, 0, 0, 0)`), unlike
                                          Matplotlib's default. Both arguments
                                          can be either single values or
                                          sequences.
                                        - `zorder`: the order in which the
                                          genes are plotted, with higher values
                                          appearing on top of lower ones.

                                        Specifying `s` or `c`/`color`/`norm`/
                                        `vmin`/`vmax` will raise an error,
                                        since these arguments conflict with the
                                        `upregulated_size` and
                                        `upregulated_color` arguments,
                                        respectively.
            downregulated_scatter_kwargs: a dictionary of keyword arguments to
                                          be passed to
                                          [`ax.scatter()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html)
                                          for downregulated genes; see the
                                          documentation of the
                                          `upregulated_scatter_kwargs` argument
                                          for details
            non_significant_scatter_kwargs: a dictionary of keyword arguments
                                            to be passed to
                                            [`ax.scatter()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html)
                                            for non-significant genes; see the
                                            documentation of the
                                            `upregulated_scatter_kwargs`
                                            argument for details
            legend: whether to add a legend showing the marker style for
                    upregulated, downregulated, and non-significant points
            legend_kwargs: a dictionary of keyword arguments to be passed to
                           [`ax.legend()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html)
                           to modify the legend, such as:

                           - `loc`, `bbox_to_anchor`, and `bbox_transform` to
                             set its location.
                           - `prop`, `fontsize`, and `labelcolor` to set its
                             font properties
                           - `facecolor` and `framealpha` to set its background
                             color and transparency
                           - `frameon=True` or `edgecolor` to add or color
                             its border. `frameon` defaults to `False`, instead
                             of Matplotlib's default of `True`.
                           - `title` to add a legend title

                           Can only be specified when `legend=True`.
            title: the title of the plot, or `None` to not add a title
            title_kwargs: a dictionary of keyword arguments to be passed to
                          [`ax.set_title()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_title.html)
                          to control text properties, such as `color` and
                          `size`. Can only be specified when `title` is not
                          `None`.
            xlabel: the x-axis label, or `None` to not label the x-axis
            xlabel_kwargs: a dictionary of keyword arguments to be passed to
                           [`ax.set_xlabel()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xlabel.html)
                           to control text properties, such as `color` and
                           `size`. Can only be specified when `xlabel` is not
                           `None`.
            ylabel: the y-axis label, or `None` to not label the y-axis.
                    Defaults to `f'$-log_{{10}}({y_column})$'`.
            ylabel_kwargs: a dictionary of keyword arguments to be passed to
                           [`ax.set_ylabel()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_ylabel.html)
                           to control text properties, such as `color` and
                           `size`. Can only be specified when `ylabel` is not
                           `None`.
            despine: whether to remove the top and right spines (borders of the
                     plot area) from the volcano plot
            savefig_kwargs: a dictionary of keyword arguments to be passed to
                            [`plt.savefig()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html),
                            such as:

                            - `dpi`: defaults to 300 instead of Matplotlib's
                              default of 150
                            - `bbox_inches`: the bounding box of the portion of
                              the figure to save; defaults to 'tight' (crop out
                              any blank borders) instead of Matplotlib's
                              default of `None` (save the entire figure)
                            - `pad_inches`: the number of inches of padding to
                              add on each of the four sides of the figure when
                              saving. Defaults to 'layout' (use the padding
                              from the constrained layout engine), instead of
                              Matplotlib's default of 0.1.
                            - `transparent`: whether to save with a transparent
                              background; defaults to `True` if saving to a PDF
                              (i.e. when `filename` ends with `'.pdf'`) and
                              `False` otherwise, instead of Matplotlib's
                              default of always being `False`.

                            Can only be specified when `filename` is specified.
        """
        # Import matplotlib
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            import matplotlib.pyplot as plt
        finally:
            signal.signal(signal.SIGINT, original_sigint_handler)

        # Check that `cell_type` is a cell type in this DE object
        check_type(cell_type, 'cell_type', str, 'a string')
        if cell_type not in self.table['cell_type']:
            error_message = \
                f'cell_type {cell_type!r} is not a cell type in this DE object'
            raise ValueError(error_message)

        # If `filename` was specified, check that `ax` is `None`, that
        # `filename` is a string or `pathlib.Path`, and that its base directory
        # exists; if `filename` is `None`, make sure `savefig_kwargs` is also
        # `None`
        if filename is not None:
            if ax is not None:
                error_message = (
                    'filename and ax are mutually exclusive; if you specified '
                    'ax in order to create multiple subplots, make a separate '
                    'savefig() call once all subplots are created')
                raise ValueError(error_message)
            check_type(filename, 'filename', (str, Path),
                       'a string or pathlib.Path')
            directory = os.path.dirname(filename)
            if directory and not os.path.isdir(directory):
                error_message = (
                    f'{filename} refers to a file in the directory '
                    f'{directory!r}, but this directory does not exist')
                raise NotADirectoryError(error_message)
            filename = str(filename)
        elif savefig_kwargs is not None:
            error_message = 'savefig_kwargs must be None when filename is None'
            raise ValueError(error_message)

        # If `figure_kwargs` was specified, check that `ax` is `None`
        if figure_kwargs is not None and ax is not None:
            error_message = (
                'figure_kwargs must be None when ax is not None, since a new '
                'figure does not need to be generated when plotting onto an '
                'existing axis')
            raise ValueError(error_message)

        # Check that `x_column`, `y_column`, and `significance_column` are the
        # names of floating-point columns in `self.table`
        for column, column_name in ((x_column, 'x_column'),
                                    (y_column, 'y_column'),
                                    (significance_column,
                                     'significance_column')):
            check_type(column, column_name, str, 'a string')
            if column not in self.table:
                error_message = (
                    f'{column_name} ({column!r}) is not a column of '
                    f'self.table')
                raise ValueError(error_message)
            check_dtype(self.table[column], f'self.table[{column!r}]',
                        'floating-point')

        # If `ylabel` was left at its default of `...`, derive it from
        # `y_column`
        if ylabel is ...:
            ylabel = f'$-log_{{10}}({y_column})$'

        # Check that `threshold` is greater than 0 and less than or equal to 1
        check_type(threshold, 'threshold', (int, float),
                   'a number > 0 and ≤ 1')
        check_bounds(threshold, 'threshold', 0, 1, left_open=True)

        # Subset `self.table` to the selected cell type, and log-transform
        # `y_column`. (This cannot be done in-place, since `y_column` may be
        # the same column as `significance_column`.) Reassign `y_column` to be
        # this log10-transformed column.
        table = self.table\
            .filter(cell_type=cell_type)\
            .with_columns(_DE_log10_y_column=-pl.col(y_column).log10())
        y_column = '_DE_log10_y_column'

        # Check that `genes_to_label` is a non-negative integer, a sequence of
        # strings, or `None`. If an integer, take that many gene names (up to
        # the number of significant DE genes) with the highest (in log space)
        # `y_column` values.
        if isinstance(genes_to_label, (int, np.integer)):
            check_bounds(genes_to_label, 'genes_to_label', 0)
            label_genes = genes_to_label > 0
            if label_genes:
                top_DE_genes = table\
                    .filter(pl.col(significance_column) < threshold)\
                    .top_k(genes_to_label, by=y_column)
                x_to_label = top_DE_genes[x_column]
                y_to_label = top_DE_genes[y_column]
                genes_to_label = top_DE_genes['gene']
                # There may be no significant genes to label
                label_genes = len(genes_to_label) > 0
        else:
            label_genes = genes_to_label is not None
            if label_genes:
                genes_to_label = \
                    to_tuple_checked(genes_to_label, 'genes_to_label', str,
                                     'strings')
                genes_to_label = pl.DataFrame({'gene': genes_to_label})\
                    .join(table.select('gene', x_column, y_column),
                          on='gene', how='left', maintain_order='left')
                num_missing = genes_to_label[x_column].null_count()
                if num_missing == len(genes_to_label):
                    error_message = (
                        "none of the specified genes were found in "
                        "table['gene']")
                    raise ValueError(error_message)
                elif num_missing > 0:
                    gene = genes_to_label\
                        .filter(pl.col(x_column).is_null())['gene'][0]
                    error_message = (
                        f"one of the specified genes, {gene!r}, was not found "
                        f"in table['gene']")
                    raise ValueError(error_message)
                x_to_label = genes_to_label[x_column]
                y_to_label = genes_to_label[y_column]
                genes_to_label = genes_to_label['gene']

        # Check that `upregulated_size`, `downregulated_size`, and
        # `non_significant_size` are positive numbers
        for size, size_name in (upregulated_size, 'upregulated_size'), \
                (downregulated_size, 'downregulated_size'), \
                (non_significant_size, 'non_significant_size'):
            check_type(size, size_name, (int, float), 'a positive number')
            check_bounds(size, size_name, 0, left_open=True)

        # Check that `upregulated_color`, `downregulated_color`, and
        # `non_significant_color` are valid Matplotlib colors, and convert them
        # to hex
        if not plt.matplotlib.colors.is_color_like(upregulated_color):
            error_message = 'upregulated_color is not a valid Matplotlib color'
            raise ValueError(error_message)
        upregulated_color = plt.matplotlib.colors.to_hex(upregulated_color)
        if not plt.matplotlib.colors.is_color_like(downregulated_color):
            error_message = \
                'downregulated_color is not a valid Matplotlib color'
            raise ValueError(error_message)
        downregulated_color = plt.matplotlib.colors.to_hex(downregulated_color)
        if not plt.matplotlib.colors.is_color_like(non_significant_color):
            error_message = \
                'non_significant_color is not a valid Matplotlib color'
            raise ValueError(error_message)
        non_significant_color = \
            plt.matplotlib.colors.to_hex(non_significant_color)

        # For each of the kwargs arguments, if the argument was specified,
        # check that it is a dictionary and that all its keys are strings.
        for kwargs, kwargs_name in ((figure_kwargs, 'figure_kwargs'),
                                    (upregulated_scatter_kwargs,
                                     'upregulated_scatter_kwargs'),
                                    (downregulated_scatter_kwargs,
                                     'downregulated_scatter_kwargs'),
                                    (non_significant_scatter_kwargs,
                                     'non_significant_scatter_kwargs'),
                                    (legend_kwargs, 'legend_kwargs'),
                                    (title_kwargs, 'title_kwargs'),
                                    (xlabel_kwargs, 'xlabel_kwargs'),
                                    (ylabel_kwargs, 'ylabel_kwargs'),
                                    (savefig_kwargs, 'savefig_kwargs')):
            if kwargs is not None:
                check_type(kwargs, kwargs_name, dict, 'a dictionary')
                for key in kwargs:
                    if not isinstance(key, str):
                        error_message = (
                            f'all keys of {kwargs_name} must be strings, but '
                            f'it contains a key of type '
                            f'{type(key).__name__!r}')
                        raise TypeError(error_message)

        # Check that the three `scatter_kwargs` arguments do not contain
        # the `s` or `c`/`color`/`cmap`/`norm`/`vmin`/`vmax` keys
        for kwargs, kwargs_prefix in (
                (upregulated_scatter_kwargs, 'upregulated'),
                (downregulated_scatter_kwargs, 'downregulated'),
                (non_significant_scatter_kwargs, 'non_significant')):
            if kwargs is None:
                continue
            if 's' in kwargs:
                error_message = (
                    f"'s' cannot be specified as a key in "
                    f"{kwargs_prefix}_scatter_kwargs; specify the "
                    f"{kwargs_prefix}_size argument instead")
                raise ValueError(error_message)
            for key in 'c', 'color', 'cmap', 'norm', 'vmin', 'vmax':
                if key in kwargs:
                    error_message = (
                        f'{key!r} cannot be specified as a key in '
                        f'{kwargs_prefix}_scatter_kwargs; specify the '
                        f'{kwargs_prefix}_color argument instead')
                    raise ValueError(error_message)

        # Override the defaults for certain values of the three
        # `scatter_kwargs` arguments
        default_scatter_kwargs = dict(rasterized=True, linewidths=0,
                                      edgecolors=(0, 0, 0, 0))
        upregulated_scatter_kwargs = \
            default_scatter_kwargs | upregulated_scatter_kwargs \
            if upregulated_scatter_kwargs is not None else \
                default_scatter_kwargs
        downregulated_scatter_kwargs = \
            default_scatter_kwargs | downregulated_scatter_kwargs \
            if downregulated_scatter_kwargs is not None else \
                default_scatter_kwargs
        non_significant_scatter_kwargs = \
            default_scatter_kwargs | non_significant_scatter_kwargs \
            if non_significant_scatter_kwargs is not None else \
                default_scatter_kwargs

        # Check that `title` is a string or `None`; if `None`, check that
        # `title_kwargs` is `None` as well. Ditto for `xlabel` and `ylabel`.
        for arg, arg_name, arg_kwargs in (
                (title, 'title', title_kwargs),
                (xlabel, 'xlabel', xlabel_kwargs),
                (ylabel, 'ylabel', ylabel_kwargs)):
            if arg is not None:
                check_type(arg, arg_name, str, 'a string')
            elif arg_kwargs is not None:
                error_message = \
                    f'{arg_name}_kwargs must be None when {arg_name} is None'
                raise ValueError(error_message)

        # Check that `legend` and `despine` are Boolean, and that
        # `legend_kwargs` is `None` when `legend=False`
        check_type(legend, 'legend', bool, 'Boolean')
        check_type(despine, 'despine', bool, 'Boolean')
        if not legend and legend_kwargs is not None:
            error_message = 'legend_kwargs must be None when legend=False'
            raise ValueError(error_message)

        # Check the label-layout parameters. `attraction`, `repulsion`,
        # `padding`, `box_padding`, `fontsize`, and `linewidth` are positive
        # numbers; `max_iter` and `seed` are non-negative integers; and
        # `linecolor` is a Matplotlib color.
        for value, value_name in ((attraction, 'attraction'),
                                  (repulsion, 'repulsion'),
                                  (padding, 'padding'),
                                  (box_padding, 'box_padding'),
                                  (fontsize, 'fontsize'),
                                  (linewidth, 'linewidth')):
            check_type(value, value_name, (int, float), 'a positive number')
            check_bounds(value, value_name, 0, left_open=True)
        for value, value_name in ((max_iter, 'max_iter'), (seed, 'seed')):
            check_type(value, value_name, (int, np.integer),
                       'a non-negative integer')
            check_bounds(value, value_name, 0)
        if not plt.matplotlib.colors.is_color_like(linecolor):
            error_message = 'linecolor is not a valid Matplotlib color'
            raise ValueError(error_message)

        # The label-layout parameters may only be set when genes are being
        # labeled; otherwise they must be left at their defaults
        if not label_genes:
            for value, default, name in (
                    (attraction, 5e-2, 'attraction'),
                    (repulsion, 1e-6, 'repulsion'),
                    (max_iter, 4000, 'max_iter'),
                    (seed, 0, 'seed'),
                    (padding, 4.0, 'padding'),
                    (box_padding, 4.0, 'box_padding'),
                    (fontsize, 10, 'fontsize'),
                    (linewidth, 0.6, 'linewidth'),
                    (linecolor, '0.45', 'linecolor')):
                if value != default:
                    error_message = (
                        f'{name} can only be specified when genes are being '
                        f'labeled, but genes_to_label is 0 or None, or there '
                        f'are no significant genes to label')
                    raise ValueError(error_message)

        # If `ax` is `None`, create a new figure; otherwise, check that it is a
        # Matplotlib axis
        fig = None
        try:
            if ax is None:
                default_figure_kwargs = dict(layout='constrained')
                figure_kwargs = default_figure_kwargs | figure_kwargs \
                    if figure_kwargs is not None else default_figure_kwargs
                fig = plt.figure(**figure_kwargs)
                ax = fig.gca()
            else:
                check_type(ax, 'ax', plt.Axes, 'a Matplotlib axis')

            # Make the volcano plot
            ax.scatter(*table
                       .filter(pl.col(significance_column) < threshold,
                               pl.col(x_column) > 0)
                       .select(x_column, y_column)
                       .to_numpy()
                       .T, s=upregulated_size, c=upregulated_color,
                       label='Upregulated', **upregulated_scatter_kwargs)
            ax.scatter(*table
                       .filter(pl.col(significance_column) < threshold,
                               pl.col(x_column) < 0)
                       .select(x_column, y_column)
                       .to_numpy()
                       .T, s=downregulated_size, c=downregulated_color,
                       label='Downregulated', **downregulated_scatter_kwargs)
            ax.scatter(*table
                       .filter(pl.col(significance_column) >= threshold)
                       .select(x_column, y_column)
                       .to_numpy()
                       .T, s=non_significant_size, c=non_significant_color,
                       label='Non-significant',
                       **non_significant_scatter_kwargs)
            ax.set_ylim(bottom=0)

            # Add the legend; override the defaults for certain values of
            # `legend_kwargs`
            if legend:
                default_legend_kwargs = dict(frameon=False)
                legend_kwargs = default_legend_kwargs | legend_kwargs \
                    if legend_kwargs is not None else default_legend_kwargs
                ax.legend(**legend_kwargs)

            # Add the title and axis labels
            if xlabel is not None:
                if xlabel_kwargs is None:
                    xlabel_kwargs = {}
                ax.set_xlabel(xlabel, **xlabel_kwargs)
            if ylabel is not None:
                if ylabel_kwargs is None:
                    ylabel_kwargs = {}
                ax.set_ylabel(ylabel, **ylabel_kwargs)
            if title is not None:
                if title_kwargs is None:
                    title_kwargs = {}
                ax.set_title(title, **title_kwargs)

            # Add labels, using our label repulsion algorithm to avoid overlap
            if label_genes:
                import_cython({'labels': 'label'})
                label(ax=ax, x=x_to_label.to_numpy(), y=y_to_label.to_numpy(),
                      texts=genes_to_label,
                      x_scatter=table[x_column].to_numpy(),
                      y_scatter=table[y_column].to_numpy(),
                      attraction=attraction, repulsion=repulsion,
                      max_iter=max_iter, seed=seed, padding=padding,
                      box_padding=box_padding, fontsize=fontsize,
                      linewidth=linewidth, linecolor=linecolor)

            # Despine, if specified
            if despine:
                spines = ax.spines
                spines['top'].set_visible(False)
                spines['right'].set_visible(False)

            # Save; override the defaults for certain keys of `savefig_kwargs`
            if filename is not None:
                default_savefig_kwargs = \
                    dict(dpi=300, bbox_inches='tight', pad_inches='layout',
                         transparent=filename.endswith('.pdf'))
                savefig_kwargs = default_savefig_kwargs | savefig_kwargs \
                    if savefig_kwargs is not None else default_savefig_kwargs
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    fig.savefig(filename, **savefig_kwargs)
                plt.close(fig)
        except:
            # If we made a new figure, make sure to close it if there's an
            # exception (but not if there was no error and `filename` is
            # `None`, in case the user wants to modify it further before
            # saving)
            if fig is not None:
                plt.close(fig)
            raise