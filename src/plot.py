import matplotlib as mpl # needed to avoid conflicts with vtk
import matplotlib.pyplot as plt

import numpy as np
import time

import itertools

from helpers import *


time_plot = 0.0
time_plot_save = 0.0


class Plot:
    def __init__(self):
        self._series = []
        self._fig = None
        self._plot_xdata_by_file_and_series = []
        self._plot_ydata_by_file_and_series = []
        self._labels_by_series = []
        self._output_files = []
        self._axis_ids = []

    def plot_to_file(self, *args):
        self._plot_to_file(*args)


    def logplot_to_file(self, *args):
        def cb(plt, ax):
            ax.set_yscale("log")
        self._plot_to_file(*(args + (cb,)))


    def _init(self, nplots, xlabel):
        height = 6 + 4*(nplots-1)
        self._fig, self._axes = plt.subplots(nplots, sharex=False, figsize=(10, height), dpi=72)
        if nplots == 1: self._axes = [axes]

        self._axes[0].set_title("created at {0}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        self._axes[-1].set_xlabel(xlabel)
        
        bot = 0.1 * 6 / height
        sep = 0.1 if nplots == 1 else 0.1 / (nplots-1)
        sep += (nplots - 1) * bot
        self._fig.subplots_adjust(
                hspace = sep,
                bottom = bot,
                top    = 1.0 - 0.05 * 6 / height,
                left=.1, right=.7)


    def add_data(self, meta, recs, outfn):
        start_time = time.clock()

        self._output_files.append(outfn)

        if isinstance(recs, list):
            recs = np.asarray(recs)

        if len(recs.shape) == 1:
            # make 2D column vector out of recs
            recs = np.expand_dims(recs, axis=1)
            assert not isinstance(meta, list)
            meta = [ meta ]

        if meta is None:
            if recs.shape[1] > 1:
                # first column to x-axis, all other columns to y axes
                meta = [ None ] * recs.shape[1]
                meta[0] = Meta(None, DoV.TIM, "x")
                for i in range(1, len(meta)):
                    meta[i] = Meta(None, DoV.VAL, "y{0}".format(i))
            else:
                # only one column, x-axis will be record number
                meta = [ Meta(None, DoV.VAL, "y") ]

        elif isinstance(meta, list) or isinstance(meta, MetaList):
            assert len(meta) == recs.shape[1]

            meta = list(meta) # make a copy

            if recs.shape[1] > 1:
                # first column to x-axis, all other columns to y axes
                if not isinstance(meta[0], Meta):
                    for i, m in enumerate(meta):
                        if i==0:
                            meta[i] = Meta(None, DoV.TIM, m)
                        else:
                            meta[i] = Meta(None, DoV.VAL, m)
            else:
                # only one column, x-axis will be record number
                if not isinstance(meta[0], Meta):
                    for i, m in enumerate(meta):
                        meta[i] = Meta(None, DoV.VAL, m)

        else:
            raise TypeError("parameter meta is neither None nor list")

        meta_by_attr = {}

        # each attribute will be plotted in a separate subplot
        times = None
        xlabel = None
        for i, m in enumerate(meta):
            a = m.attr
            if m.dov == DoV.TIM:
                times = recs[:, i]
                xlabel = a
            elif m.dov == DoV.VAL:
                if a not in meta_by_attr:
                    meta_by_attr[a] = [(i, m)]
                else:
                    meta_by_attr[a].append((i, m))
        if recs.shape[1] == 1:
            xlabel = "n"
        else:
            # times = [ i for i, _ in enumerate(recs) ]
            assert times is not None

        self._xlabel = xlabel
        self._meta_by_attr = meta_by_attr

        only_update_data = not not self._labels_by_series
        if not only_update_data:
            nplots = len(meta_by_attr)
            self._ymins = [ float("+inf") for i in range(nplots) ]
            self._ymaxs = [ float("-inf") for i in range(nplots) ]

        series_id = 0
        plot_xdata_by_series = []
        plot_ydata_by_series = []
        self._plot_xdata_by_file_and_series.append(plot_xdata_by_series)
        self._plot_ydata_by_file_and_series.append(plot_ydata_by_series)

        for ax_id, am in enumerate(sorted(meta_by_attr.items())):
            attr = am[0]
            metas = am[1]

            ymax = float("-inf")
            ymin = float("inf")
            for i, m in metas:
                ys = recs[:,i]
                # TODO why shouldn't there be ys?
                if len(ys) > 0:
                    ymax = max(ymax, max(ys))
                    ymin = min(ymin, min(ys))

                    if times is not None:
                        plot_xdata_by_series.append(times)
                    else:
                        plot_xdata_by_series.append(None)
                    plot_ydata_by_series.append(ys)
                    if not only_update_data: self._labels_by_series.append(m)

                    self._axis_ids.append(ax_id)
                    series_id += 1
            self._ymaxs[ax_id] = max(self._ymaxs[ax_id], ymax)
            self._ymins[ax_id] = min(self._ymins[ax_id], ymin)


    def do_plots(self):
        start_time = time.clock()

        nplots = len(self._plot_xdata_by_file_and_series[0])
        self._init(nplots, self._xlabel)

        markers = [ 'x', '+' ]

        series_id = 0

        # if (not only_update_data) and style_cb is not None: style_cb(plt, ax)

        first_file = True
        for fn, xdata_by_series, ydata_by_series in zip(
                self._output_files,
                self._plot_xdata_by_file_and_series,
                self._plot_ydata_by_file_and_series):

            for series_id, (xdata, ydata, label, ax_id) in enumerate(zip(
                xdata_by_series, ydata_by_series,
                self._labels_by_series, self._axis_ids)):

                ax = self._axes[ax_id]

                marker = itertools.cycle(markers)

                if xdata is not None:
                    # no markers if more than 50 data points
                    ma = marker.next() if len(ydata) <= 50 else None
                    if not first_file:
                        assert len(self._series) > series_id
                        ser = self._series[series_id]
                        ser.set_xdata(xdata)
                        ser.set_ydata(ydata)
                    else:
                        assert series_id == len(self._series)
                        self._series.append(ax.plot(xdata, ydata, label=label, marker=ma, markersize=5)[0])
                else:
                    # no markers if more than 50 data points
                    ma = marker.next() if len(ydata) <= 50 else None
                    if not first_file:
                        assert len(self._series) > series_id
                        ser = self._series[series_id]
                        ser.set_ydata(ydata)
                    else:
                        assert series_id == len(self._series)
                        self._series.append(ax.plot(ydata, label=label, marker=ma, markersize=5)[0])

            if first_file:
                for ax, ymin, ymax in zip(self._axes, self._ymins, self._ymaxs):
                    # adaptively switch to log scale
                    if ymax > ymin and ymin > 0.0:
                        if math.log10(ymax/ymin) > 2:
                            ax.set_yscale('log')

                    ax.set_ylim(ymin, ymax)

                    ax.grid()

                    font_p = mpl.font_manager.FontProperties()
                    font_p.set_size("small")
                    ax.legend(loc="upper left", bbox_to_anchor=(1,1), prop = font_p)


            start_time_save = time.clock()
            if isinstance(fn, str):
                self._fig.savefig(fn)
            else:
                fmt = os.path.basename(outfh.name).split(".")[-1]
                self._fig.savefig(fn, format=fmt)
            global time_plot_save
            time_plot_save += time.clock() - start_time_save

            first_file = False

        global time_plot
        time_plot += time.clock() - start_time


    def __del__(self):
        if self._fig:
            plt.close(self._fig)




