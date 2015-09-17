
import datetime
import numbers
import math

class DoV:
    TIM = 0
    DOM = 1
    VAL = 2

    def __init__(self, *args):
        pass

    def __str__(self):
        if   self == DoV.TIM: return "tim"
        elif self == DoV.DOM: return "dom"
        elif self == DoV.VAL: return "val"
        raise ValueError("unrecognized DoV constant")

    @staticmethod
    def from_str(s):
        if   s == "tim": return DoV.TIM
        elif s == "dom": return DoV.DOM
        elif s == "val": return DoV.VAL
        raise ValueError("unrecognized DoV constant: {0}".format(s))

DoV.TIM = DoV(DoV.TIM)
DoV.DOM = DoV(DoV.DOM)
DoV.VAL = DoV(DoV.VAL)


# check objects for equality in a generic way
class EqMixin(object):
    def __eq__(self, other):
        return type(other) is type(self) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self.__eq__(other)


class Meta(EqMixin):
    def __init__(self, src, dov=None, attr=None, comp=None, pt_or_elem_id=None, tfm=False):
        if src and isinstance(src, Meta):
            # copy other
            self.col  = src.col
            self.dov  = src.dov
            self.pex  = src.pex
            self.attr = src.attr
            self.comp = src.comp
            self.src  = src.src
            self.tfm  = src.tfm
        elif src and isinstance(src, dict):
            self.col  = int(src["c"]) - 1 if "c" in src else None
            self.dov  = DoV.from_str(src["t"])
            self.pex  = src["pe"] if "pe" in src else None
            self.attr = src["a"]
            self.comp = src["cmp"] if "cmp" in src else None
            self.src  = src["src"] if "src" in src else None
            self.tfm  = src["tfm"] if "tfm" in src else False
        else:
            assert isinstance(dov, DoV)
            assert attr != None

            self.col  = None
            self.dov  = dov
            self.pex  = pt_or_elem_id
            self.attr = attr
            self.comp = comp
            self.src  = src
            self.tfm  = tfm

    def __str__(self):
        s = ""
        if self.attr is not None:
            s += self.attr
        if self.comp is not None:
            s += "[{0}]".format(self.comp)
        if self.pex is not None:
            s += " at {0}".format(self.pex)
        if self.src is not None:
            if self.tfm:
                s += " ({0}, transformed)".format(self.src)
            else:
                s += " ({0})".format(self.src)
        elif self.tfm:
            s += " (transformed)"

        return s # ATTR[COMP] at pt PT (SRC, transformed)

    def __unicode__(self):
        return unicode(str(self))

    def __iter__(self):
        attrs = []
        if self.col != None: attrs.append(("c", self.col))
        attrs.append(("t", self.dov))
        if self.pex != None: attrs.append(("pe", self.pex))
        attrs.append(("a", self.attr))
        if self.comp != None: attrs.append(("cmp", self.comp))
        if self.src != None: attrs.append(("src", self.src))
        if self.tfm: attrs.append(("tfm", self.tfm))

        for k, v in attrs:
            yield (k, v)

    def get_attr_id(self):
        return "attr {0} comp {1} at pt {2}".format(self.attr, self.pex, self.comp)


class MetaList(EqMixin):
    def __init__(self, metas):
        self.ms = metas

    # get all columns where the given keywords have the respective values
    def get_columns(self, **kwargs):
        cols = list(self.columns(**kwargs))
        # this assertion makes using this method's output as array index safer.
        # TODO difference numpy index empty tuple vs. empty list
        assert cols # didn't find any column matching the given selector
        return cols

    def get_column(self, **kwargs):
        cols = self.get_columns(**kwargs)
        assert len(cols) == 1
        return cols[0]

    def get_column_from(self, recs, **kwargs):
        return recs[:, self.get_column(**kwargs)]

    def get_columns_from(self, recs, **kwargs):
        return recs[:, self.get_columns(**kwargs)]

    def columns(self, **kwargs):
        for i, m in enumerate(self.ms):
            for k, v in kwargs.items():
                a = getattr(m, k)
                if isinstance(v, str):
                    if not fnmatchcase(a, v):
                        break
                elif a != v:
                    break
            else:
                # yield only if match
                yield i

    # record (prop_value, column_id) for each value of property prop
    # filtered by kwargs
    def each(self, prop, **kwargs):
        map_prop_cols = {}
        for ci in self.columns(**kwargs):
            pval = getattr(self.ms[ci], prop)
            if pval not in map_prop_cols:
                map_prop_cols[pval] = []
            map_prop_cols[pval].append(ci)

        return sorted(map_prop_cols.items())

    def __getitem__(self, i):
        return self.ms[i]

    def __iter__(self):
        return self.ms.__iter__()

    def __len__(self):
        return len(self.ms)

    def append(self, *args, **kwargs):
        self.ms.append(*args, **kwargs)


class Cell(EqMixin):
    def __init__(self, i):
        self.value = int(i)

    def get(self):
        return self.value

    def flatten(self): return None

    def get_x_value(self): return None

    def __str__(self):
        return "cell {}".format(self.value)

# TODO add property "x-value"
class Point(EqMixin):
    def __init__(self, s, x_value=None):
        self.index = -1
        self.coords = []
        self.x_values = [ x_value ]

        if isinstance(s, basestring):
            self.init_string(s)
        elif isinstance(s, numbers.Integral):
            self.index = s
        else:
            self.coords = [ list(s) ]

    def get(self):
        return self.index

    def get_coords(self):
        return self.coords

    def get_x_value(self):
        return self.x_values[0]

    def __str__(self):
        if self.coords:
            return "pt ({})".format(", ".join(str(x) for x in self.coords[0]))
        else:
            return "pt {}".format(self.index)

    def flatten(self):
        if len(self.coords) <= 1: return None
        
        return [ Point(c, x) for c, x in zip(self.coords, self.x_values) ]

    def init_string(self, s):
        try:
            index = int(s)
            assert index >= 0
            self.index = index
        except ValueError:
            parts = s.split("/")
            if len(parts) == 1:
                self.index = -1
                self.coords = [ self.parse_coords(s) ]

            elif len(parts) == 3:
                self.index = -1
                c1 = self.parse_coords(parts[0])
                c2 = self.parse_coords(parts[2])
                assert len(c1) == len(c2)

                diff = math.sqrt(sum( (x1-x2)**2 for x1, x2 in zip(c1, c2) ))
                assert diff != 0

                self.coords = []
                self.x_values = []

                delta = parts[1].strip()
                if delta[0] == '#':
                    delta = int(delta[1:]) # delta is number of equally spaced points
                    assert delta > 1
                    for i in range(delta):
                        t = float(i) / (delta-1)
                        cs = list(c1)
                        for ci in range(len(cs)):
                            cs[ci] = (1.0 - t) * c1[ci] + t * c2[ci]
                        self.coords.append(cs)
                        self.x_values.append(diff*t)

                else:
                    # make evenly distributed point with distance of delta
                    delta = float(delta)
                    assert delta > 0

                    i=0
                    while i*delta < diff:
                        t = i*delta/diff
                        cs = list(c1)
                        for ci in range(len(cs)):
                            cs[ci] = (1.0 - t) * c1[ci] + t * c2[ci]
                        self.coords.append(cs)
                        self.x_values.append(i*delta)

                        i=i+1
            else:
                assert False # value has wrong format

    @staticmethod
    def parse_coords(s):
        coords = [ float(p) for p in s.split(",") ]
        assert len(coords) != 0 and len(coords) <= 3
        return coords


