#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import codecs


class TextGrid(object):
    def __init__(
        self,
        file_type="",
        object_class="",
        xmin=0.0,
        xmax=0.0,
        tiers_status="",
        tiers=[],
    ):
        self.file_type = file_type
        self.object_class = object_class
        self.xmin = xmin
        self.xmax = xmax
        self.tiers_status = tiers_status
        self.tiers = tiers

        if self.xmax < self.xmin:
            raise ValueError("xmax ({}) < xmin ({})".format(self.xmax, self.xmin))

    def cutoff(self, xstart=None, xend=None):
        if xstart is None:
            xstart = self.xmin

        if xend is None:
            xend = self.xmax

        if xend < xstart:
            raise ValueError("xend ({}) < xstart ({})".format(xend, xstart))

        new_xmax = xend - xstart + self.xmin
        new_xmin = self.xmin
        new_tiers = []

        for tier in self.tiers:
            new_tiers.append(tier.cutoff(xstart=xstart, xend=xend))
        return TextGrid(
            file_type=self.file_type,
            object_class=self.object_class,
            xmin=new_xmin,
            xmax=new_xmax,
            tiers_status=self.tiers_status,
            tiers=new_tiers,
        )


class Tier(object):
    def __init__(self, tier_class="", name="", xmin=0.0, xmax=0.0, intervals=[]):
        self.tier_class = tier_class
        self.name = name
        self.xmin = xmin
        self.xmax = xmax
        self.intervals = intervals

        if self.xmax < self.xmin:
            raise ValueError("xmax ({}) < xmin ({})".format(self.xmax, self.xmin))

    def cutoff(self, xstart=None, xend=None):
        if xstart is None:
            xstart = self.xmin

        if xend is None:
            xend = self.xmax

        if xend < xstart:
            raise ValueError("xend ({}) < xstart ({})".format(xend, xstart))

        bias = xstart - self.xmin
        new_xmax = xend - bias
        new_xmin = self.xmin
        new_intervals = []
        for interval in self.intervals:
            if interval.xmax <= xstart or interval.xmin >= xend:
                pass
            elif interval.xmin < xstart:
                new_intervals.append(
                    Interval(
                        xmin=new_xmin, xmax=interval.xmax - bias, text=interval.text
                    )
                )
            elif interval.xmax > xend:
                new_intervals.append(
                    Interval(
                        xmin=interval.xmin - bias, xmax=new_xmax, text=interval.text
                    )
                )
            else:
                new_intervals.append(
                    Interval(
                        xmin=interval.xmin - bias,
                        xmax=interval.xmax - bias,
                        text=interval.text,
                    )
                )

        return Tier(
            tier_class=self.tier_class,
            name=self.name,
            xmin=new_xmin,
            xmax=new_xmax,
            intervals=new_intervals,
        )


class Interval(object):
    def __init__(self, xmin=0.0, xmax=0.0, text=""):
        self.xmin = xmin
        self.xmax = xmax
        self.text = text

        if self.xmax < self.xmin:
            raise ValueError("xmax ({}) < xmin ({})".format(self.xmax, self.xmin))


def read_textgrid_from_file(filepath):
    with codecs.open(filepath, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    if lines[-1] == "\r\n":
        lines = lines[:-1]

    assert "File type" in lines[0], "error line 0, {}".format(lines[0])
    file_type = (
        lines[0]
        .split("=")[1]
        .replace(" ", "")
        .replace('"', "")
        .replace("\r", "")
        .replace("\n", "")
    )

    assert "Object class" in lines[1], "error line 1, {}".format(lines[1])
    object_class = (
        lines[1]
        .split("=")[1]
        .replace(" ", "")
        .replace('"', "")
        .replace("\r", "")
        .replace("\n", "")
    )

    assert lines[2] == "\r\n", "error line 2, {}".format(lines[2])

    assert "xmin" in lines[3], "error line 3, {}".format(lines[3])
    xmin = float(
        lines[3].split("=")[1].replace(" ", "").replace("\r", "").replace("\n", "")
    )

    assert "xmax" in lines[4], "error line 4, {}".format(lines[4])
    xmax = float(
        lines[4].split("=")[1].replace(" ", "").replace("\r", "").replace("\n", "")
    )

    assert "tiers?" in lines[5], "error line 5, {}".format(lines[5])
    tiers_status = (
        lines[5].split("?")[1].replace(" ", "").replace("\r", "").replace("\n", "")
    )

    assert "size" in lines[6], "error line 6, {}".format(lines[6])
    size = int(
        lines[6].split("=")[1].replace(" ", "").replace("\r", "").replace("\n", "")
    )

    assert lines[7] == "item []:\r\n", "error line 7, {}".format(lines[7])

    tier_start = []
    for item_idx in range(size):
        tier_start.append(lines.index(" " * 4 + "item [{}]:\r\n".format(item_idx + 1)))

    tier_end = tier_start[1:] + [len(lines)]

    tiers = []
    for tier_idx in range(size):
        tiers.append(
            read_tier_from_lines(
                tier_lines=lines[tier_start[tier_idx] + 1 : tier_end[tier_idx]]
            )
        )

    return TextGrid(
        file_type=file_type,
        object_class=object_class,
        xmin=xmin,
        xmax=xmax,
        tiers_status=tiers_status,
        tiers=tiers,
    )


def read_tier_from_lines(tier_lines):
    assert "class" in tier_lines[0], "error line 0, {}".format(tier_lines[0])
    tier_class = (
        tier_lines[0]
        .split("=")[1]
        .replace(" ", "")
        .replace('"', "")
        .replace("\r", "")
        .replace("\n", "")
    )

    assert "name" in tier_lines[1], "error line 1, {}".format(tier_lines[1])
    name = (
        tier_lines[1]
        .split("=")[1]
        .replace(" ", "")
        .replace('"', "")
        .replace("\r", "")
        .replace("\n", "")
    )

    assert "xmin" in tier_lines[2], "error line 2, {}".format(tier_lines[2])
    xmin = float(
        tier_lines[2].split("=")[1].replace(" ", "").replace("\r", "").replace("\n", "")
    )

    assert "xmax" in tier_lines[3], "error line 3, {}".format(tier_lines[3])
    xmax = float(
        tier_lines[3].split("=")[1].replace(" ", "").replace("\r", "").replace("\n", "")
    )

    assert "intervals: size" in tier_lines[4], "error line 4, {}".format(tier_lines[4])
    intervals_num = int(
        tier_lines[4].split("=")[1].replace(" ", "").replace("\r", "").replace("\n", "")
    )

    # handle unformatted case
    # R12_S203204205_C09_I1_Near_203.TextGrid
    # R12_S203204205_C09_I1_Near_205.TextGrid
    if tier_lines[-1] == "\n":
        tier_lines = tier_lines[:-1]

    if len(tier_lines[5:]) == intervals_num * 5:
        intervals = []
        for intervals_idx in range(intervals_num):
            assert tier_lines[
                5 + 5 * intervals_idx + 0
            ] == " " * 8 + "intervals [{}]:\r\n".format(intervals_idx + 1)
            assert tier_lines[
                5 + 5 * intervals_idx + 1
            ] == " " * 8 + "intervals [{}]:\r\n".format(intervals_idx + 1)
            intervals.append(
                read_interval_from_lines(
                    interval_lines=tier_lines[
                        7 + 5 * intervals_idx : 10 + 5 * intervals_idx
                    ]
                )
            )
    elif len(tier_lines[5:]) == intervals_num * 4:
        # handle unformatted case
        # R12_S203204205_C09_I1_Near_203.TextGrid
        # R12_S203204205_C09_I1_Near_204.TextGrid
        # R12_S203204205_C09_I1_Near_205.TextGrid
        intervals = []
        for intervals_idx in range(intervals_num):
            assert tier_lines[
                5 + 4 * intervals_idx + 0
            ] == " " * 8 + "intervals [{}]:\r\n".format(intervals_idx + 1)

            intervals.append(
                read_interval_from_lines(
                    interval_lines=tier_lines[
                        6 + 4 * intervals_idx : 9 + 4 * intervals_idx
                    ]
                )
            )
    else:
        import pdb

        pdb.set_trace()
        raise ValueError(
            "error lines {} % {} != 0".format(len(tier_lines[5:]), intervals_num)
        )

    return Tier(
        tier_class=tier_class, name=name, xmin=xmin, xmax=xmax, intervals=intervals
    )


def read_interval_from_lines(interval_lines):
    assert len(interval_lines) == 3, "error lines"

    assert "xmin" in interval_lines[0], "error line 0, {}".format(interval_lines[0])
    xmin = float(
        interval_lines[0]
        .split("=")[1]
        .replace(" ", "")
        .replace("\r", "")
        .replace("\n", "")
    )

    assert "xmax" in interval_lines[1], "error line 1, {}".format(interval_lines[1])
    xmax = float(
        interval_lines[1]
        .split("=")[1]
        .replace(" ", "")
        .replace("\r", "")
        .replace("\n", "")
    )

    assert "text" in interval_lines[2], "error line 2, {}".format(interval_lines[2])
    text = (
        interval_lines[2]
        .split("=")[1]
        .replace(" ", "")
        .replace('"', "")
        .replace("\r", "")
        .replace("\n", "")
    )

    return Interval(xmin=xmin, xmax=xmax, text=text)
