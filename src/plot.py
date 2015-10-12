# -*- encoding: utf-8 -*-

import matplotlib as mpl # needed to avoid conflicts with vtk
import matplotlib.pyplot as plt

import numpy as np
import time

import Queue
import itertools

import multiprocessing
import subprocess
import threading

from helpers import *

import os.path
import sys


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
        
        self._work_queue = multiprocessing.Queue()

    def plot_to_file(self, *args):
        self._plot_to_file(*args)


    def logplot_to_file(self, *args):
        def cb(plt, ax):
            ax.set_yscale("log")
        self._plot_to_file(*(args + (cb,)))


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

        meta_by_attr = {} # (attr, axis) --> [ (index, meta), ... ]
        meta_by_attr_order = []

        # each attribute will be plotted in a separate subplot
        times = None
        xlabel = None
        for i, m in enumerate(meta):
            if m.dov == DoV.TIM:
                times = recs[:, i]
                xlabel = m.attr
            elif m.dov == DoV.VAL:
                key = (m.attr if m.axis == -1 else None, m.axis)
                if key not in meta_by_attr:
                    meta_by_attr[key] = [(i, m)]
                    meta_by_attr_order.append(key)
                else:
                    meta_by_attr[key].append((i, m))
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

        for ax_id, key in enumerate(meta_by_attr_order):
            metas = meta_by_attr[key]

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
                    if not only_update_data:
                        self._labels_by_series.append(m)
                        self._axis_ids.append(ax_id)
                    series_id += 1
            self._ymaxs[ax_id] = max(self._ymaxs[ax_id], ymax)
            self._ymins[ax_id] = min(self._ymins[ax_id], ymin)


class MPLPlot(Plot):
    def _init(self, nplots, xlabel):
        height = 6 + 4*(nplots-1)
        fig, axes = plt.subplots(nplots, sharex=False, figsize=(10, height), dpi=72)
        if nplots == 1: axes = [axes]

        axes[0].set_title("created at {0}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        axes[-1].set_xlabel(xlabel)
        
        bot = 0.1 * 6 / height
        sep = 0.1 if nplots == 1 else 0.1 / (nplots-1)
        sep += (nplots - 1) * bot
        fig.subplots_adjust(
                hspace = sep,
                bottom = bot,
                top    = 1.0 - 0.05 * 6 / height,
                left=.1, right=.7)

        return fig, axes


    def _do_plot(self, worker_id, fig, axes):
        first_file = True

        markers = [ 'x', '+' ]

        while True:
            work = self._work_queue.get()
            if work is None:
                self._work_queue.put(None)
                break

            fn, xdata_by_series, ydata_by_series = work

            # print(worker_id, "fn", fn)

            for series_id, (xdata, ydata, label, ax_id) in enumerate(zip(
                xdata_by_series, ydata_by_series,
                self._labels_by_series, self._axis_ids)):

                ax = axes[ax_id]

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
                for ax, ymin, ymax in zip(axes, self._ymins, self._ymaxs):
                    # adaptively switch to log scale
                    if ymax > ymin and ymin > 0.0:
                        if math.log10(ymax/ymin) > 2:
                            ax.set_yscale('log')

                    if ymax > ymin:
                        d = ymax - ymin
                        ymax += 0.025 * d
                        ymin -= 0.025 * d
                        ax.set_ylim(ymin, ymax)

                    ax.grid()

                    font_p = mpl.font_manager.FontProperties()
                    font_p.set_size("small")
                    ax.legend(loc="upper left", bbox_to_anchor=(1,1), prop = font_p)


            start_time_save = time.clock()
            if isinstance(fn, str):
                fig.savefig(fn)
            else:
                fmt = os.path.basename(fn.name).split(".")[-1]
                fig.savefig(fn, format=fmt)
            # global time_plot_save
            # time_plot_save += time.clock() - start_time_save

            first_file = False


    def do_plots(self, num_threads=1):
        if num_threads == 0:
            num_threads = multiprocessing.cpu_count()
        if num_threads > 1:
            print("plotting data using {} threads".format(num_threads))

        start_time = time.time()

        nplots = len(self._plot_xdata_by_file_and_series[0])

        # if (not only_update_data) and style_cb is not None: style_cb(plt, ax)
        
        for fn, xdata_by_series, ydata_by_series in zip(
                self._output_files,
                self._plot_xdata_by_file_and_series,
                self._plot_ydata_by_file_and_series):
            self._work_queue.put((fn, xdata_by_series, ydata_by_series))
        self._work_queue.put(None)

        workers = []
        figs = []
        for i in range(num_threads):
            fig, axes = self._init(nplots, self._xlabel)
            figs.append(fig)
            w = multiprocessing.Process(target=self._do_plot, args=(i, fig, axes))
            workers.append(w)
            w.start()

        for w, fig in zip(workers, figs):
            w.join()
            plt.close(fig)

        global time_plot
        time_plot += time.time() - start_time


# TODO xlabel, markers, log scale
class GnuPlot(Plot):
    def _write_plot(self, gp, xdatas, ydatas, labels, markers):
        first = True

        gp("set lmargin 9\n")

        for label, marker, xdata in zip(labels, markers, xdatas):
            if first:
                first = False
                plotcmd = 'plot "-"'
            else:
                plotcmd = ', ""'

            if xdata is None:
                if marker:
                    gp('{} u 0:1 w lp t "{}"'.format(plotcmd, label.short_format()))
                else:
                    gp('{} u 0:1 w l t "{}"'.format(plotcmd, label.short_format()))
            else:
                if marker:
                    gp('{} w lp t "{}"'.format(plotcmd, label.short_format()))
                else:
                    gp('{} w l t "{}"'.format(plotcmd, label.short_format()))
        gp("\n")

        for xdata, ydata in zip(xdatas, ydatas):
            if xdata is None:
                for y in ydata:
                    # precision is chosen s.t. both steps and jitter are avoided
                    gp(" {:.14e}\n".format(y))
                gp("EOF\n")
            else:
                for x, y in zip(xdata, ydata):
                    gp(" {:.14e} {:.14e}\n".format(x, y))
                gp("EOF\n")

    def _write_data_to_gnuplot(self, worker_id, proc):
        # nplots = len(self._plot_xdata_by_file_and_series[0])
        nplots = len(np.unique(self._axis_ids))
        xplots = 1 if nplots <= 6 else 2
        yplots = (nplots+1) // xplots
        width = 800 if xplots == 1 else 1200
        height=width/8/xplots*(6+4*(yplots-1))

        def debug_gp(w):
            def wrt(s):
                sys.stderr.write(s)
                w(s)
            return wrt

        # gp = debug_gp(proc.stdin.write)
        gp = proc.stdin.write
        gp("set encoding utf8\n")
        gp("set terminal pngcairo noenhanced size {},{} \n".format(width, height))

        gp("""
set grid back
# set key noenhanced
# set autoscale fix
# set offsets 0, 0, graph 0.025, graph 0.025
set border back
set tics in back
# set xzeroaxis ls -1
set mxtics
# set mytics
""")

        while True:
            work = self._work_queue.get()
            if work is None:
                self._work_queue.put(None)
                break

            fn, xdata_by_series, ydata_by_series = work

            gp("set output \"{}\"\n".format(fn))
            title = "{} – created at {}".format(
                    os.path.basename(fn),
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            gp('set multiplot layout {}, {} title "{}"\n'.format(yplots, xplots, title))

            # gather data by axes
            for ax_id in np.unique(self._axis_ids):

                xdatas = []
                ydatas = []
                labels = []
                mas = []

                for series_id in np.where(np.array(self._axis_ids) == ax_id)[0]:
                    xdata = xdata_by_series[series_id]
                    ydata = ydata_by_series[series_id]
                    label = self._labels_by_series[series_id]

                    # no markers if more than 50 data points
                    ma = marker.next() if len(ydata) <= 50 else None

                    xdatas.append(xdata)
                    ydatas.append(ydata)
                    labels.append(label)
                    mas.append(ma)

                self._write_plot(gp, xdatas, ydatas, labels, mas)

            gp("unset multiplot\n")
            gp("set output\n")
        proc.stdin.write("exit\n")
        proc.stdin.close()

    def do_plots(self, num_threads=1):
        if num_threads == 0:
            num_threads = multiprocessing.cpu_count()
        if num_threads > 1:
            print("plotting data using {} threads".format(num_threads))

        start_time = time.time()

        # if (not only_update_data) and style_cb is not None: style_cb(plt, ax)
        
        for fn, xdata_by_series, ydata_by_series in zip(
                self._output_files,
                self._plot_xdata_by_file_and_series,
                self._plot_ydata_by_file_and_series):
            self._work_queue.put((fn, xdata_by_series, ydata_by_series))
        self._work_queue.put(None)

        workers = []
        procs = []
        for i in range(num_threads):
            proc = subprocess.Popen(["gnuplot"], stdin=subprocess.PIPE)
            procs.append(proc)

            w = threading.Thread(target=self._write_data_to_gnuplot, args=(i, proc))
            workers.append(w)

            w.start()

        for w, p in zip(workers, procs):
            w.join()
            p.wait()

        global time_plot
        time_plot += time.time() - start_time
