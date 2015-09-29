#!/usr/bin/python
#
# Copyright (C) 2015 Christoph Lehmann
#
# This file is part of pvd-tool.
#
# pvd-tool is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pvd-tool is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pvd-tool.  If not, see <http://www.gnu.org/licenses/>.
#


# TODO
# * more metadata written to csv (timestamp, cmdline args, ...)


import sys
import argparse
import os

import xml.etree.cElementTree as ET

import numpy as np

import re

from collections import OrderedDict
import json

import imp

import time # for performance measurement

from fnmatch import fnmatchcase

import numbers
import six

import plot
from helpers import *

# Plot = plot.MPLPlot
Plot = plot.GnuPlot

time_total = time.time()
time_import_vtk = 0.0


def die(msg, status=1):
    sys.stderr.write(msg)
    sys.exit(status)


def warn(msg):
    sys.stderr.write("WARNING: {0}\n".format(msg))



class JsonSer(json.JSONEncoder):
    def default(self, o):
        try:
            iterable = iter(o)
            return OrderedDict(iterable)
        except TypeError:
            if isinstance(o, (DoV, Point, Cell)):
                return str(o)

        return json.JSONEncoder.default(self, o)



def getFilesTimes(xmlTree, pathroot):
    node = xmlTree.getroot()
    if node.tag != "VTKFile": return None, None
    children = list(node)
    if len(children) != 1: return None, None
    node = children[0]
    if node.tag != "Collection": return None, None

    ts = []
    fs = []

    for child in node:
        if child.tag != "DataSet": return None, None
        ts.append(float(child.get("timestep")))
        fs.append(relpathfrom(pathroot, child.get("file")))

    return ts, fs


def relpathfrom(origin, relpath):
    if os.path.isabs(relpath):
        return relpath
    return os.path.join(origin, relpath)


# TODO allow for wildcard attributes
# returns: list of (index, name)
def get_attribute_idcs(fieldData, attrs):
    idcs = []
    idcs_set = set()
    for a in attrs:
        found = False
        num_arr = fieldData.GetNumberOfArrays()
        for i in xrange(num_arr):
            n = fieldData.GetArray(i).GetName()
            if fnmatchcase(n, a):
                if i not in idcs_set:
                    idcs.append((i, n))
                    idcs_set.add(i)
                found = True
        # if num_arr != 0 and not found:
        #     warn("Attribute %s not found" % a)
    return idcs



def apply_script(fcts, timesteps, grids):
    assert len(timesteps) == len(grids)
    res = [ None for _ in range(len(grids)) ]
    for i in xrange(len(grids)):
        ts = timesteps[i]
        grid = grids[i]
        ngrid = vtk.vtkUnstructuredGrid()
        ngrid.DeepCopy(grid)

        # TODO extend for cells
        gridPoints = ngrid.GetPoints()
        numPt = gridPoints.GetNumberOfPoints()

        gridPD = ngrid.GetPointData()
        for ai in xrange(gridPD.GetNumberOfArrays()):
            arr = gridPD.GetArray(ai)
            attr = arr.GetName()

            for pi in xrange(numPt):
                coords = gridPoints.GetPoint(pi)
                tup = arr.GetTuple(pi)
                if attr in fcts:
                    ntup = fcts[attr](ts, coords)
                    if type(ntup) == float: ntup = (ntup,)
                    assert len(tup) == len(ntup)
                else:
                    warn("no function found for attribute {}".format(attr))
                    ntup = None

                arr.SetTuple(pi, ntup)

        res[i] = ngrid

    return res


def get_point_data_from_grid(
        point, point_id, grid, src, attrIdcs, attrData, incl_coords,
        rec, meta
        ):
    gridPoints = grid.GetPoints()
    npts = gridPoints.GetNumberOfPoints()
    
    if point_id >= npts or point_id < 0:
        warn("point index {} out of bounds [0,{}]\n".format(point_id, npts-1))
        return

    if incl_coords:
        coords = gridPoints.GetPoint(point_id)
        for ci, coord in enumerate(coords):
            rec.append(coord)
            meta.append(Meta(src, DoV.DOM, "coord", ci, point))

    for ai, a in enumerate(attrData):
        an = attrIdcs[ai][1]
        comps = a.GetTuple(point_id)
        for ci, comp in enumerate(comps):
            comp = comps[ci]
            rec.append(comp)
            meta.append(Meta(src, DoV.VAL, an, ci, point))

def get_cell_data_from_grid(
        cell, grid, src, attrIdcs, attrData, incl_coords,
        rec, meta
        ):
    gridCells = grid.GetCells()
    ncells = gridCells.GetNumberOfCells()
    
    c = cell.get()
    if c >= ncells or c < 0:
        warn("{} out of bounds [0,{}]\n".format(cell, ncells-1))
        return

    if incl_coords:
        # add dummy coordinates
        coords = grid.GetPoints().GetPoint(0)
        for ci, _ in enumerate(coords):
            rec.append(0.0)
            meta.append(Meta(src, DoV.DOM, "coord", ci, cell))

    for ai, a in enumerate(attrData):
        an = attrIdcs[ai][1]
        comps = a.GetTuple(c)
        for ci, comp in enumerate(comps):
            comp = comps[ci]
            rec.append(comp)
            meta.append(Meta(src, DoV.VAL, an, ci, cell))


def filter_grid_ts(src, grid, timestep, attrs, points_cells, incl_coords):
    gridPoints = grid.GetPoints()
    gridCells  = grid.GetCells()

    attrIdcsPt = get_attribute_idcs(grid.GetPointData(), attrs)
    attrDataPt = [ grid.GetPointData().GetArray(i) for i, _ in attrIdcsPt ]

    attrIdcsCell = get_attribute_idcs(grid.GetCellData(), attrs)
    attrDataCell = [ grid.GetCellData().GetArray(i) for i, _ in attrIdcsCell ]

    npts = gridPoints.GetNumberOfPoints()
    ncells = gridCells.GetNumberOfCells()

    if (npts + ncells) > 0:
        # categorize points: index or coordinates
        coord_pts = []
        map_point_indices = {} # maps point index in the list to point index in probeFilter

        for i, point_cell in enumerate(points_cells):
            if isinstance(point_cell, Point):
                coords = point_cell.get_coords()
                if coords:
                    map_point_indices[i] = len(coord_pts)
                    coord_pts.append(coords)

        if coord_pts:
            interpPts = vtk.vtkPoints()
            for c in coord_pts:
                interpPts.InsertNextPoint(*c)

            interpData = vtk.vtkPolyData()
            interpData.SetPoints(interpPts)

            probeFilter = vtk.vtkProbeFilter()
            probeFilter.SetSourceData(grid)
            probeFilter.SetInputData(interpData)
            probeFilter.Update()

            grid_interpolated = probeFilter.GetOutput()
            attrIdcsCoords = get_attribute_idcs(grid_interpolated.GetPointData(), attrs)
            attrDataCoords = [ grid_interpolated.GetPointData().GetArray(i) for i, _ in attrIdcsCoords ]

        rec = []
        meta = []

        rec.append(timestep)
        meta.append(Meta(src, DoV.TIM, "time"))

        for i, point_cell in enumerate(points_cells):
            if isinstance(point_cell, Point):
                if point_cell.get_coords():
                    p = map_point_indices[i]
                    get_point_data_from_grid(point_cell, p, grid_interpolated, src, attrIdcsCoords, attrDataCoords, incl_coords,
                        rec, meta)
                else:
                    p = point_cell.get()
                    get_point_data_from_grid(point_cell, p, grid, src, attrIdcsPt, attrDataPt, incl_coords,
                        rec, meta)
            elif isinstance(point_cell, Cell):
                get_cell_data_from_grid(point_cell, grid, src, attrIdcsCell, attrDataCell, incl_coords,
                        rec, meta)
            else:
                print("Error: Given object is neither point nor cell index")
                assert False

        return rec, MetaList(meta)
    
    return None, None


def filter_grid_dom(src, grid, attrs, points_cells, incl_coords):
    gridPoints = grid.GetPoints()
    gridCells  = grid.GetCells()

    attrIdcsPt = get_attribute_idcs(grid.GetPointData(), attrs)
    attrDataPt = [ grid.GetPointData().GetArray(i) for i, _ in attrIdcsPt ]

    attrIdcsCell = get_attribute_idcs(grid.GetCellData(), attrs)
    attrDataCell = [ grid.GetCellData().GetArray(i) for i, _ in attrIdcsCell ]

    npts = gridPoints.GetNumberOfPoints()
    ncells = gridCells.GetNumberOfCells()

    if points_cells is None:
        points_cells = [ Point(i) for i in range(npts) ]

    if (npts + ncells) > 0:
        # categorize points: index or coordinates
        coord_pts = []
        map_point_indices = {} # maps point index in the list to point index in probeFilter

        for i, point_cell in enumerate(points_cells):
            if isinstance(point_cell, Point):
                coords = point_cell.get_coords()
                if coords:
                    map_point_indices[i] = len(coord_pts)
                    coord_pts.append(coords)

        if coord_pts:
            interpPts = vtk.vtkPoints()
            for c in coord_pts:
                interpPts.InsertNextPoint(*c)

            interpData = vtk.vtkPolyData()
            interpData.SetPoints(interpPts)

            probeFilter = vtk.vtkProbeFilter()
            probeFilter.SetSourceData(grid)
            probeFilter.SetInputData(interpData)
            probeFilter.Update()

            grid_interpolated = probeFilter.GetOutput()
            attrIdcsCoords = get_attribute_idcs(grid_interpolated.GetPointData(), attrs)
            attrDataCoords = [ grid_interpolated.GetPointData().GetArray(i) for i, _ in attrIdcsCoords ]

        recs = []
        meta = []
        meta.append(Meta(src, DoV.TIM, "ordinal number or dist from first point"))

        first_loop = True

        for i, point_cell in enumerate(points_cells):
            x = point_cell.get_x_value()
            if x is None: x = i
            rec = [ x ]
            tmp_meta = []
            if isinstance(point_cell, Point):
                if point_cell.get_coords():
                    p = map_point_indices[i]
                    get_point_data_from_grid(point_cell, p, grid_interpolated, src, attrIdcsCoords,
                            attrDataCoords, incl_coords, rec, tmp_meta)
                else:
                    p = point_cell.get()
                    get_point_data_from_grid(point_cell, p, grid, src, attrIdcsPt,
                            attrDataPt, incl_coords, rec, tmp_meta)
            elif isinstance(point_cell, Cell):
                get_cell_data_from_grid(point_cell, grid, src, attrIdcsCell, attrDataCell, True,
                        rec, tmp_meta)
            else:
                print("Error: Given object is neither point nor cell index")
                assert False

            if first_loop:
                first_loop = False
                meta.extend(tmp_meta)
            recs.append(rec)

        return recs, MetaList(meta)
    
    return None, None

def filter_grid_dom_old(src, grid, attrs):
    gridPoints = grid.GetPoints()

    attrIdcs = get_attribute_idcs(grid.GetPointData(), attrs)
    attrData = [ grid.GetPointData().GetArray(i) for i, _ in attrIdcs ]

    npts = gridPoints.GetNumberOfPoints()

    meta = []
    recs = []

    first_loop = True

    for p in xrange(gridPoints.GetNumberOfPoints()):
        rec = []

        coords = gridPoints.GetPoint(p)
        for ci in xrange(len(coords)):
            coord = coords[ci]
            rec.append(coord)
            if first_loop: meta.append(Meta(src, DoV.DOM, "coord %i" % ci))

        for ai in xrange(len(attrData)):
            a = attrData[ai]
            an = attrIdcs[ai][1]
            comps = a.GetTuple(p)
            for ci in xrange(len(comps)):
                comp = comps[ci]
                rec.append(comp)
                if first_loop: meta.append(Meta(src, DoV.VAL, "%s[%i]" % (an, ci)))

        first_loop = False
        recs.append(rec)

    return recs, MetaList(meta)


def write_pvd(outfh, timesteps, vtus):
    assert len(timesteps) == len(vtus)
    outfh.write('<?xml version="1.0"?>\n'
                '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian" compressor="vtkZLibDataCompressor">\n'
                '  <Collection>\n')

    for i in range(len(timesteps)):
        outfh.write('    <DataSet timestep="{0}" group="" part="0" file="{1}"/>\n'.format(timesteps[i], vtus[i]))

    outfh.write("  </Collection>\n"
                "</VTKFile>")
    outfh.close()


def write_csv(meta, records, outFile, precision, json_enc):
    header = "Columns:\n"
    header2 = "\n"

    nc = len(meta)
    header += "[\n"
    old_meta = None
    for i in xrange(nc):
        # TODO: more tabular format for header
        if old_meta and (
                old_meta.src != meta[i].src or old_meta.dov != meta[i].dov
                or old_meta.pex != meta[i].pex or old_meta.tfm != meta[i].tfm
                ): header += "\n"

        meta[i].col = "{0:2}".format(i+1)
        header += "  {0}".format(json_enc.encode(meta[i]))
        if i != nc-1: header += ","
        header += "\n"

        old_meta = meta[i]

        # if i == 0:
        #     header2 += "{{:>{}}}".format(precision+5).format(meta[i].attr)
        # else:
        if i != 0:
            header2 += " "
            colwidth = precision + 7
        else:
            colwidth = precision + 5
        attr = meta[i].attr
        if len(attr) > colwidth:
            attr = attr[:colwidth-2] + ".."
        header2 += "{{:>{}}}".format(colwidth).format(attr)

    header += "]\n" + header2

    np.savetxt(outFile, records,
            delimiter=" ",
            fmt="%{}.{}g".format(precision+7, precision),
            header=header)


def read_csv(fh, parse_header=True):
    if isinstance(fh, six.string_types):
        with open(fh) as fh_:
            return read_csv(fh_, parse_header)

    meta = None

    if parse_header:
        mode = 0 # initial
        json_str = ""

        while True:
            lastpos = fh.tell()
            line = fh.readline()
            if not line: break

            if line.startswith("#"):
                line = line.lstrip("#").lstrip()
                if mode == 0:
                    if line.startswith("Columns:"):
                        mode = 1
                elif mode == 1: # "Columns:" in previous line
                    if line.rstrip() == "[":
                        mode = 2
                        json_str += line
                    elif not line:
                        # ignore empty line
                        pass
                    else:
                        warn("Unexpected header format. I will not attempt to process it.")
                        break
                elif mode == 2: # assemble json
                    json_str += line
                    if line.rstrip() == "]":
                        break
            elif not line.strip():
                # ignore empty line
                pass
            else:
                # no comment line
                warn("unexpected end of header. Json found so far:\n{0}".format(json))
                json_str = None
                fh.seek(lastpos)
                break

        if json:
            meta = MetaList(json.loads(json_str, object_hook=Meta))

    arr = np.loadtxt(fh)

    return arr, meta


def gather_files(infh):
    if isinstance(infh, str):
        fn = infh
    else:
        fn = infh.name

    if fn.endswith(".pvd"):
        pathroot = os.path.dirname(fn)
        pcdtree = ET.parse(infh)
        timesteps, files = getFilesTimes(pcdtree, pathroot)
    elif fn.endswith(".vtu"):
        timesteps = [0]
        files = [ fn ]
    else:
        die("File `%s' has unknown type" % fn)

    return timesteps, files


def gather_grids(infh, reader, filefilter=None):
    def get_grid(path):
        reader.SetFileName(path)
        reader.Update()
        g = vtk.vtkUnstructuredGrid()
        g.DeepCopy(reader.GetOutput())
        return g

    timesteps, fs = gather_files(infh)

    grids = [ None ] * len(timesteps)
    for i, (f, t) in enumerate(zip(fs, timesteps)):
        if (not filefilter) or filefilter.filter(t, f):
            grids[i] = get_grid(f)

    return timesteps, grids


def get_timeseries(src, grids, tss, attrs, points, incl_coords):
    oldMeta = None
    records = []

    for i in xrange(len(grids)):
        rec, meta = filter_grid_ts(src, grids[i], tss[i], attrs, points, incl_coords)
        if rec is not None:
            if oldMeta is None:
                oldMeta = meta
            else:
                assert meta == oldMeta
            records.append(rec)

    return records, oldMeta


def get_point_data(src, grids, attrs, points_cells, output_coords):
    oldMeta = None
    records = []
    meta = []

    for i, g in enumerate(grids):
        if g:
            recs, meta = filter_grid_dom(src, g, attrs, points_cells, output_coords)
            if oldMeta is None:
                oldMeta = meta
            else:
                assert meta == oldMeta
            records.append(recs)
        else:
            records.append(())

    return records, meta


def combine_arrays(arrays):
    if len(arrays) == 1: return arrays[0]

    res = []
    nr = len(arrays[0])
    na = len(arrays)

    for ri in range(nr):
        row = []
        for ai in range(na):
            assert len(arrays[ai]) == nr
            row += arrays[ai][ri]
        res.append(row)
    return res


def combine_domains(metas, recs):
    ncols = len(metas)

    nmetas = []
    nrecs = []
    first_row = True

    for row in recs:
        assert len(row) == ncols
        lbls = {}
        nrow = []

        for ci in range(ncols):
            m = metas[ci]
            val = row[ci]

            if m.dov != DoV.VAL:
                lbl = m.get_attr_id()
                if lbl in lbls:
                    assert val == row[lbls[lbl]]
                else:
                    lbls[lbl] = ci
                    nrow.append(val)
                    if first_row:
                        nmeta = Meta(m)
                        nmeta.src = None
                        nmetas.append(nmeta)
            else:
                nrow.append(val)
                if first_row:
                    nmeta = Meta(m)
                    nmetas.append(nmeta)

        first_row = False
        nrecs.append(nrow)

    return nmetas, nrecs


# argparse types

def InputFile(val):
    parts = val.split(":", 2)

    if len(parts) == 2:
        try:
            if parts[1] == "-":
                fh = sys.stdin
            else:
                path = os.path.expanduser(parts[1])
                assert os.path.isfile(path) and os.access(path, os.R_OK)
                fh = path
        except IOError:
            warn("Warning: Could not open `{0}', will try `{1}' instead".format(parts[1], val))
        else:
            return parts[0], fh

    if val == "-":
        fh = sys.stdin
    else:
        try:
            path = os.path.expanduser(val)
            assert os.path.isfile(path) and os.access(path, os.R_OK)
            fh = path
        except AssertionError as e:
            raise argparse.ArgumentTypeError("input file `{}' is not readable or not a file".format(path))
        return None, fh


def DirectoryW(val):
    # TODO implement
    return val


def InputFileArgument(path):
    path = os.path.expanduser(path)
    assert os.path.isfile(path) and os.access(path, os.R_OK)
    return path

def OutputFileArgument(path):
    path = os.path.expanduser(path)
    assert os.path.isfile(path) and os.access(path, os.W_OK)
    return path

re_out_file = re.compile(r'^([%@^][0-9]+)+:')
def OutputFile(val):
    m = re_out_file.match(val)
    # if not m: raise argparse.ArgumentTypeError("`{0}' does not correspond to the output file path format".format(val))
    if m:
        path = val[m.end():]
        tfm_and_num = val[m.start():m.end()-1]
    else:
        # TODO maybe add info message
        path = val
        tfm_and_num = "^0"

    try:
        if path == "-":
            outfh = sys.stdout
        else:
            path = os.path.expanduser(path)
            assert os.path.isfile(path) and os.access(path, os.W_OK)
            outfh = path
    except IOError as e:
        raise argparse.ArgumentTypeError("I/O error({0}) when trying to open `{2}': {1}".format(e.errno, e.strerror, path))

    
    spl = re.split(r'([%@^])', tfm_and_num)
    assert len(spl) % 2 == 1 # empty string at the beginning, then pairs of [%@^] and a number

    nums_tfms = []

    for i in range(1, len(spl), 2):
        tfm_char = spl[i]
        if tfm_char == '^': do_transform = 0
        if tfm_char == '@': do_transform = 1
        if tfm_char == '%': do_transform = 2

        nums_tfms.append((int(spl[i+1]), do_transform))

    return (nums_tfms, outfh)


def OutputDir(val):
    m = re_out_file.match(val)
    # if not m: raise argparse.ArgumentTypeError("`{0}' does not correspond to the output directory path format".format(val))
    if m:
        path = val[m.end():]
        tfm_and_num = val[m.start():m.end()-1]
    else:
        # TODO maybe add info message
        path = val
        tfm_and_num = "^0"

    path = os.path.expanduser(path)
    d = os.path.dirname(path) or "."
    if not os.path.isdir(d):
        raise argparse.ArgumentTypeError("`{0}' is not a directory".format(d))
    
    spl = re.split(r'([%@^])', tfm_and_num)
    assert len(spl) % 2 == 1 # empty string at the beginning, then pairs of [%@^] and a number

    nums_tfms = []

    for i in range(1, len(spl), 2):
        tfm_char = spl[i]
        if tfm_char == '^': do_transform = 0
        if tfm_char == '@': do_transform = 1
        if tfm_char == '%': do_transform = 2

        nums_tfms.append((int(spl[i+1]), do_transform))

    return (nums_tfms, path)


def check_consistency_ts(args):
    assert args.points_cells is not None and len(args.points_cells) != 0 # at least one point or cell must be chosen

    for nums_tfms, _ in args.out_csv or []:
        for num, tfm in nums_tfms:
            assert num < len(args.in_files)
            assert args.script or not tfm # if script is used, script must be given

    for nums_tfms, _ in args.out_plot or []:
        for num, tfm in nums_tfms:
            assert num < len(args.in_files)
            assert args.script or not tfm # if script is used, script must be given


def check_consistency_dom(args):
    # assert (not args.out_pvd) != (not args.attr)
    
    if args.points_cells:
        t = None
        for pc in args.points_cells:
            if t is not None:
                assert type(pc) is t # either only points or only cells are allowed
            else:
                t = type(pc)

    for nums_tfms, _ in args.out_csv or []:
        # assert len(nums_tfms) == 1 # currently no combination of whole grids allowed
        for num, tfm in nums_tfms:
            assert num < len(args.in_files)
            assert args.script or not tfm # if script is used, script must be given

    for nums_tfms, _ in args.out_pvd or []:
        assert len(nums_tfms) == 1 # currently no combination of whole grids allowed
        for num, tfm in nums_tfms:
            assert num < len(args.in_files)
            assert args.script or not tfm # if script is used, script must be given


def load_input_files(in_files, req_out, script_fh, script_params, filefilter):
    assert len(in_files) > 0

    # check that all input files are of the same type (either vtu or pvd)
    input_type = None
    for _, f in in_files:
        path = f if isinstance(f, six.string_types) else f.name
        if not path: continue
        m = re.search(r'[.][^.]*$', path)
        if m:
            if input_type is None:
                input_type = m.group(0)
            elif input_type != m.group(0):
                print("Error: You must not mix input files of different type!")
                assert input_type == m.group(0)

    if script_fh is not None and isinstance(script_fh, list): script_fh = script_fh[0]
    reader = vtk.vtkXMLUnstructuredGridReader()

    timesteps = [ None for _ in range(len(in_files)) ]
    vtuFiles =  [ None for _ in range(len(in_files)) ]
    vtuFiles_transformed = [ None for _ in range(len(in_files)) ]

    scr_loaded = False

    if input_type == ".pvd":
        timesteps = [ None for _ in range(len(in_files)) ]
        vtuFiles =  [ None for _ in range(len(in_files)) ]
        vtuFiles_transformed = [ None for _ in range(len(in_files)) ]

        # load and, if necessary, transform source files
        for nums_tfms, _ in req_out:
            for num, tfm in nums_tfms:
                if not vtuFiles[num]:
                    timesteps[num], vtuFiles[num] = gather_grids(in_files[num][1], reader, filefilter)
                if tfm != 0:
                    assert script_fh is not None
                    if not scr_loaded:
                        script_args = {}
                        for kv in script_params:
                            k, v = kv.split('=', 2)
                            script_args[k] = v

                        analytical_model = imp.load_source("analytical_model", script_fh.name, script_fh)
                        analytical_model.init(script_args)
                        scr_loaded = True
                    if not vtuFiles_transformed[num]:
                        vtuFiles_transformed[num] = apply_script(analytical_model.get_attribute_functions(), timesteps[num], vtuFiles[num])
    elif input_type == ".vtu":
        timesteps = [ [ None ]*len(in_files) ]
        vtuFiles =  [ [ None ]*len(in_files) ]
        vtuFiles_transformed = [ None ]

        # load and, if necessary, transform source files
        for nums_tfms, _ in req_out:
            for num, tfm in nums_tfms:
                assert num == 0
                for fi, (_, in_file) in enumerate(in_files):
                    if filefilter.filter(fi, in_file):
                        _, vtu = gather_grids(in_file, reader)
                        timesteps[0][fi] = fi
                        vtuFiles[0][fi] = vtu[0]
                if tfm != 0:
                    assert script_fh is not None
                    if not scr_loaded:
                        script_args = {}
                        for kv in script_params:
                            k, v = kv.split('=', 2)
                            script_args[k] = v

                        analytical_model = imp.load_source("analytical_model", script_fh.name, script_fh)
                        analytical_model.init(script_args)
                        scr_loaded = True
                    if not vtuFiles_transformed[0]:
                        vtuFiles_transformed[0] = apply_script(analytical_model.get_attribute_functions(), timesteps[0], vtuFiles[0])

    return timesteps, vtuFiles, vtuFiles_transformed


def get_output_data_diff(aggr_data, req_out):
    for nums_tfms, outfh in req_out:
        meta_attr_comp = {}
        meta = []
        recs = []
        for num, tfm in nums_tfms:
            assert tfm == 0
            if   tfm == 0: rng = [0]
            elif tfm == 1: rng = [1]
            elif tfm == 2: rng = [0,1]
            for tfm_idx in rng:
                r, m = aggr_data[num][tfm_idx]
                recs.append(r)
                meta += m

                for mt in m:
                    if mt.dov != DoV.VAL: continue
                    a = mt.attr
                    c = mt.comp
                    if a not in meta_attr_comp: meta_attr_comp[a] = set()
                    meta_attr_comp[a].add(c)

        meta = MetaList(meta)
        recs = combine_arrays(recs)

        for attr, comps in sorted(meta_attr_comp.items()):
            print("{} -- {}".format(attr, ", ".join([str(c) for c in comps])))

            for comp in comps:
                cols = meta.get_columns(attr=attr, comp=comp, dov=DoV.VAL)
                if len(cols) < 2:
                    warn("attribute {}[{}] is only present in one input file. skipping".format(attr, comp))
                    continue
                assert len(cols) == 2

                c0 = cols[0]
                c1 = cols[1]

                meta.append(Meta(None, DoV.VAL, attr + "_diff", comp))
                meta.append(Meta(None, DoV.VAL, attr + "_reldiff", comp))

                for r in recs:
                    v0 = r[c0]
                    v1 = r[c1]

                    diff = v0-v1
                    r += [diff, diff / max(abs(v0), abs(v1))]


        # for attr, cols in meta.each("attr", dov=DoV.VAL):
        #     print("{} -- {}".format(attr, ", ".join([str(c) for c in cols])))

        yield meta, recs, outfh


class FileFilterByTimestep:
    def __init__(self, timesteps):
        if timesteps:
            self._timesteps = sorted([ float(t) for t in timesteps ])
        else:
            self._timesteps = None

    def filter(self, ts, fn):
        if self._timesteps:
            for t in self._timesteps:
                print("ts vs t {} {} -- {} ?<? {}".format(ts, t, abs(ts-t), sys.float_info.epsilon))
                if abs(ts-t) < sys.float_info.epsilon \
                        or (ts != 0.0 and abs(ts-t)/ts < 1.e-6):
                    return True
        else:
            return True


# split_re matches any number, possibly in scientific notation
split_re = re.compile(r'([+-]?[0-9]+(?:[.][0-9]+)?(?:[eE][+-]?[0-9]+)?)')

# returns a sorted version of the given list like `sort -V`
def version_sort(in_files):
    return sorted(in_files, key=lambda f: [
        s if i%2==0 else float(s)
        for i, s in enumerate(split_re.split(
            f[1] if isinstance(f[1], six.string_types) else f[1].name
            ))
        ])


# TODO provide a similar function also for similar cases
def process_timeseries_diff(args):
    if not args.attr: args.attr = ['*']
    if args.out_plot:
        import matplotlib as mpl # needed to avoid conflicts with vtk
        import matplotlib.pyplot as plt
        globals()["mpl"] = mpl
        globals()["plt"] = plt

    # has to be imported after matplotlib
    import vtk
    globals()["vtk"] = vtk

    in_files = args.in_files
    if args.version_sort:
        in_files = version_sort(in_files)

    assert len(args.points_cells) == 1 # currently only one point at once

    if args.out_csv:
        # output file uses both input files and not transforms
        args.out_csv = [ ([(0, 0), (1, 0)], fh) for fh in args.out_csv ]
    if args.out_plot:
        # output file uses both input files and not transforms
        args.out_plot = [ ([(0, 0), (1, 0)], fh) for fh in args.out_plot ]

    req_out = (args.out_csv or []) \
            + (args.out_plot or [])
    assert len(req_out) > 0

    timesteps, vtuFiles, vtuFiles_transformed = \
            load_input_files(in_files, req_out, None, None)

    # aggregate timeseries data
    aggr_data = [ [ None, None ] for _ in in_files ]

    for nums_tfms, _ in req_out:
        for num, tfm in nums_tfms:
            assert tfm == 0 # no transformations allowed here
            src = in_files[num][0]
            if src is None: src = in_files[num][1].name

            tss = timesteps[num]
            if   tfm == 0: rng = [0]
            elif tfm == 1: rng = [1]
            elif tfm == 2: rng = [0,1]

            for tfm_idx in rng:
                if aggr_data[num][tfm_idx]: continue

                if tfm_idx != 0:
                    grids = vtuFiles_transformed[num]
                else:
                    grids = vtuFiles[num]

                # TODO find better solution for out_coords
                recs, meta = get_timeseries(src, grids, tss, args.attr, args.points_cells, args.out_coords)
                if tfm_idx != 0:
                    for m in meta: m.tfm = True
                aggr_data[num][tfm_idx] = (recs, meta)

    if args.out_csv:
        json_enc = JsonSer()
        for meta, recs, outfh in get_output_data_diff(aggr_data, args.out_csv):
            if True: #args.combine_domains:
                meta, recs = combine_domains(meta, recs)

            write_csv(meta, recs, outfh, args.csv_prec[0], json_enc)

    if args.out_plot:
        plt = Plot()
        for meta, recs, outfh in get_output_data_diff(aggr_data, args.out_plot):
            plt.plot_to_file(meta, recs, outfh)



def process_timeseries(args):
    if not args.attr: args.attr = ['*']
    if args.out_plot:
        import matplotlib as mpl # needed to avoid conflicts with vtk
        import matplotlib.pyplot as plt
        globals()["mpl"] = mpl
        globals()["plt"] = plt

    # has to be imported after matplotlib
    import vtk
    globals()["vtk"] = vtk

    check_consistency_ts(args)

    # there shall be only single points or cells in the list
    points_cells = []
    for pc in args.points_cells:
        pc_flat = pc.flatten()
        if pc_flat:
            points_cells.extend(pc_flat)
        else:
            points_cells.append(pc)

    in_files = args.in_files
    if args.version_sort:
        in_files = version_sort(in_files)

    req_out = (args.out_csv or []) \
            + (args.out_plot or [])
    assert len(req_out) > 0

    timesteps, vtuFiles, vtuFiles_transformed = \
            load_input_files(in_files, req_out, args.script, args.script_param)

    # aggregate timeseries data
    aggr_data = [ [ None, None ] for _ in in_files ]

    for nums_tfms, _ in req_out:
        for num, tfm in nums_tfms:
            src = in_files[num][0]
            if src is None: src = in_files[num][1].name

            tss = timesteps[num]
            if   tfm == 0: rng = [0]
            elif tfm == 1: rng = [1]
            elif tfm == 2: rng = [0,1]

            for tfm_idx in rng:
                if aggr_data[num][tfm_idx]: continue

                if tfm_idx != 0:
                    grids = vtuFiles_transformed[num]
                else:
                    grids = vtuFiles[num]

                recs, meta = get_timeseries(src, grids, tss, args.attr, points_cells, args.out_coords)
                if tfm_idx != 0:
                    for m in meta: m.tfm = True
                aggr_data[num][tfm_idx] = (recs, meta)

    # write csv files
    json_enc = JsonSer()

    for nums_tfms, outfh in args.out_csv or []:
        meta = []
        recs = []
        for num, tfm in nums_tfms:
            if   tfm == 0: rng = [0]
            elif tfm == 1: rng = [1]
            elif tfm == 2: rng = [0,1]
            for tfm_idx in rng:
                r, m = aggr_data[num][tfm_idx]
                recs.append(r)
                meta += m
        recs = combine_arrays(recs)

        if args.combine_domains:
            meta, recs = combine_domains(meta, recs)

        write_csv(meta, recs, outfh, args.csv_prec[0], json_enc)


    # plot
    plt = Plot()
    for nums_tfms, outfh in args.out_plot or []:
        meta = []
        recs = []
        for num, tfm in nums_tfms:
            if   tfm == 0: rng = [0]
            elif tfm == 1: rng = [1]
            elif tfm == 2: rng = [0,1]
            for tfm_idx in rng:
                r, m = aggr_data[num][tfm_idx]
                recs.append(r)
                meta += m
        recs = combine_arrays(recs)

        plt.plot_to_file(meta, recs, outfh)


def process_whole_domain(args):
    if not args.attr:     args.attr = ['*']

    # has to be imported after matplotlib
    import vtk
    globals()["vtk"] = vtk

    check_consistency_dom(args)

    if args.points_cells:
        # there shall be only single points or cells in the list
        points_cells = []
        for pc in args.points_cells:
            pc_flat = pc.flatten()
            if pc_flat:
                points_cells.extend(pc_flat)
            else:
                points_cells.append(pc)
    else:
        points_cells = None

    in_files = args.in_files
    if args.version_sort:
        in_files = version_sort(in_files)

    req_out = (args.out_csv or []) \
            + (args.out_pvd or []) \
            + (args.out_plot or [])

    timesteps, vtuFiles, vtuFiles_transformed = \
            load_input_files(in_files, req_out, args.script, args.script_param, FileFilterByTimestep(args.timestep))

    # write csv files
    json_enc = JsonSer()

    if args.out_csv or args.out_plot:
        # get data
        aggr_data = [ [ None, None ] for _ in range(len(in_files)) ]

        for nums_tfms, _ in (args.out_csv or []) + (args.out_plot or []):
            for num, tfm in nums_tfms:
                src = in_files[num][0]
                if src is None: src = in_files[num][1]

                if   tfm == 0: rng = [0]
                elif tfm == 1: rng = [1]
                elif tfm == 2:
                    rng = [0,1]
                for tfm_idx in rng:
                    if aggr_data[num][tfm_idx]: continue

                    if tfm_idx != 0:
                        grids = vtuFiles_transformed[num]
                    else:
                        grids = vtuFiles[num]

                    # TODO add switch cells/points
                    recs, meta = get_point_data(src, grids, args.attr, points_cells, args.out_coords)
                    if tfm_idx != 0:
                        for m in meta: m.tfm = True
                    aggr_data[num][tfm_idx] = (recs, meta)

        if args.out_csv:
            # write csv files
            for nums_tfms, outdirn in args.out_csv:
                for ti in range(len(timesteps[nums_tfms[0][0]])):
                    meta = []
                    recs = []
                    for num, tfm in nums_tfms:
                        assert timesteps[num] == timesteps[nums_tfms[0][0]]
                        if   tfm == 0: rng = [0]
                        elif tfm == 1: rng = [1]
                        elif tfm == 2: rng = [0,1]
                        for tfm_idx in rng:
                            r, m = aggr_data[num][tfm_idx]
                            recs.append(r[ti])
                            meta += m
                    recs = combine_arrays(recs)

                    if args.combine_domains:
                        meta, recs = combine_domains(meta, recs)

                    if recs:
                        if len(timesteps) == 1:
                            fn = outdirn \
                                    + re.sub(r"[.][^.]+$", ".csv", os.path.basename(in_files[ti][1]))
                        else:
                            t = timesteps[num][ti]
                            if isinstance(t, numbers.Integral):
                                max_ts = max(timesteps[num])
                                width = len(str(max_ts))
                                fn = ("{}_{:0"+str(width)+"}.csv").format(outdirn, t)
                            else:
                                fn = "{}_{}.csv".format(outdirn, t)
                            print("csv output to {}".format(fn))
                        write_csv(meta, recs, fn, args.csv_prec[0], json_enc)

        if args.out_plot:
            assert(args.num_threads >= 0)

            # plot
            plt = Plot()
            for nums_tfms, outdirn in args.out_plot or []:
                for ti in range(len(timesteps[nums_tfms[0][0]])):
                    meta = []
                    recs = []
                    for num, tfm in nums_tfms:
                        assert timesteps[num] == timesteps[nums_tfms[0][0]]
                        if   tfm == 0: rng = [0]
                        elif tfm == 1: rng = [1]
                        elif tfm == 2: rng = [0,1]
                        for tfm_idx in rng:
                            r, m = aggr_data[num][tfm_idx]
                            # TODO: add x-axis value
                            recs.append(r[ti])
                            meta += m
                    recs = combine_arrays(recs)

                    if recs:
                        if len(timesteps) == 1:
                            fn = outdirn \
                                    + re.sub(r"[.][^.]+$", ".png", os.path.basename(in_files[ti][1]))
                        else:
                            t = timesteps[num][ti]
                            if isinstance(t, numbers.Integral):
                                max_ts = max(timesteps[num])
                                width = len(str(max_ts))
                                fn = ("{}_{:0"+str(width)+"}.png").format(outdirn, t)
                            else:
                                fn = "{}_{}.png".format(outdirn, t)
                            print("plot output to {}".format(fn))
                        plt.add_data(meta, recs, fn)
            plt.do_plots(args.num_threads)


    # write pvd files
    if args.out_pvd:
        writer = vtk.vtkXMLUnstructuredGridWriter()

        for nums_tfms, outfh in args.out_pvd:
            outfn = outfh.name
            outf_base = re.sub(r'.pvd', '', outfn)

            out_vtus = []

            for num, tfm in nums_tfms:
                src = in_files[num][0]
                if src is None: src = in_files[num][1].name

                if   tfm == 0: rng = [0]
                elif tfm == 1: rng = [1]
                elif tfm == 2:
                    assert tfm != 2
                    rng = [0,1]
                for tfm_idx in rng:
                    if tfm_idx != 0:
                        grids = vtuFiles_transformed[num]
                    else:
                        grids = vtuFiles[num]

                    for ti in range(len(timesteps[num])):
                        # TODO: make output file names resemble input file names
                        fn = "{0}_{1}.vtu".format(outf_base, timesteps[num][ti])
                        out_vtus.append(fn)
                        writer.SetFileName(fn)
                        writer.SetInputData(grids[ti])
                        writer.Write()

            write_pvd(outfh, timesteps[num], out_vtus)


def process_proxy(args):
    script_fh = args.script[0]

    script_args = {}
    for kv in args.script_param:
        k, v = kv.split('=', 2)
        script_args[k] = v

    analytical_model = imp.load_source("analytical_model", script_fh.name, script_fh)
    analytical_model.init(script_args)
    analytical_model.proxied(args.in_files, args.out_files)


def _run_main():
    parser = argparse.ArgumentParser(description="Process PVD files")

    # common
    parser_common = argparse.ArgumentParser(description="Common options", add_help=False)

    parser_common.add_argument("-s", "--script", nargs=1,     type=InputFileArgument, help="script for generating field data, e.g., exact solutions of FEM models")
    parser_common.add_argument("--script-param", "--sp", action="append", help="parameters for the script", default=[])

    # I/O
    parser_io = argparse.ArgumentParser(description="Input/output options", add_help=False)

    parser_io.add_argument("-i", "--in", nargs='+', type=InputFile, required=True, help="input file", dest="in_files", metavar="IN_FILE")
    parser_io.add_argument("--no-combine-domains", action="store_false", dest="combine_domains", help="do not combine domains when aggregating several input files into one output file")
    parser_io.add_argument("--csv-prec", nargs=1, type=int, help="decimal precision for csv output", default=[6])
    parser_io.add_argument("--no-coords", action="store_false", dest="out_coords", help="do not output coordinate columns")
    parser_io.add_argument("--version-sort-inputs", "-V", action="store_true", dest="version_sort", help="version sort input file names before further processing")


    subparsers = parser.add_subparsers(dest="subcommand", help="subcommands")
    subparsers.required = True


    parser_frag_ts = argparse.ArgumentParser(description="compute timeseries", add_help=False)
    parser_frag_ts.add_argument("-p", "--point", type=Point, action="append", required=False, dest="points_cells")
    parser_frag_ts.add_argument("-c", "--cell",  type=Cell,  action="append", required=False, dest="points_cells")
    parser_frag_ts.add_argument("-a", "--attr",            action="append", required=False)


    # timeseries
    parser_ts = subparsers.add_parser("timeseries", help="compute timeseries", parents=[parser_io, parser_common, parser_frag_ts])
    parser_ts.set_defaults(func=process_timeseries)
    parser_ts.add_argument("--out-plot", action="append", type=OutputFile)
    parser_ts.add_argument("--out-csv",  action="append", type=OutputFile)


    # timeseries diff
    parser_tsd = subparsers.add_parser("ts-diff", help="compute differences between two timeseries", parents=[parser_frag_ts])
    parser_tsd.add_argument("-i", "--in", nargs=2, type=InputFile, required=True, help="input file", dest="in_files", metavar="IN_FILE")
    parser_tsd.add_argument("--out-plot", nargs=1, type=OutputFileArgument)
    parser_tsd.add_argument("--out-csv",  nargs=1, type=OutputFileArgument)
    parser_tsd.add_argument("--csv-prec", nargs=1, type=int, help="decimal precision for csv output", default=[6])
    parser_tsd.set_defaults(func=process_timeseries_diff)


    # domain
    parser_dom = subparsers.add_parser("domain", help="dom help", parents=[parser_io, parser_common, parser_frag_ts])

    parser_dom.add_argument("--out-pvd",           action="append", type=OutputFile)
    parser_dom.add_argument("--out-csv",           action="append", type=OutputDir)
    parser_dom.add_argument("--out-plot",          action="append", type=OutputDir)
    parser_dom.add_argument("-t", "--timestep",    action="append", required=False)
    parser_dom.add_argument("-N", "--num-threads", type=int, default=1)

    parser_dom.set_defaults(func=process_whole_domain)


    # proxy
    parser_proxy = subparsers.add_parser("proxy", help="proxy help", parents=[parser_common])
    parser_proxy.add_argument("-i", "--in", action="append", type=InputFileArgument, help="input file", dest="in_files", metavar="IN_FILE", default=[])
    parser_proxy.add_argument("-o", "--out", action="append", type=OutputFileArgument, help="output file", dest="out_files", metavar="OUT_FILE", default=[])
    parser_proxy.set_defaults(func=process_proxy)
    

    args = parser.parse_args()

    args.func(args)

    global time_total, time_import_vtk
    print("total execution took {} seconds".format(time.time() - time_total))
    print("importing vtk took   {} seconds".format(time_import_vtk))
    print("plotting took        {} seconds".format(plot.time_plot))
    print("saving plots took    {} seconds".format(plot.time_plot_save))


if __name__ == "__main__":
    _run_main()
else:
    import matplotlib as mpl # needed to avoid conflicts with vtk
    import matplotlib.pyplot as plt
    # has to be imported after matplotlib
    try:
        start_time = time.clock()
        import vtk
        time_import_vtk = time.clock() - start_time
    except ImportError:
        warn("module vtk will not be available")

