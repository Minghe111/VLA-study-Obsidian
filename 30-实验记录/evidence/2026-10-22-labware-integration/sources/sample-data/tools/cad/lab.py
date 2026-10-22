"""STEP models for the lab pack, built from our own dimensions.

Development tool only (the STEP files are committed; the seeder never runs it):

    pip install cadquery
    python tools/cad/lab.py

Each part is a revolved profile along Z (origin at the tip or orifice, the
open end at the top). Outer dimensions match the technical drawings; the
tubes are solid, the pipette tip is hollow. No threads, no text.
"""
import math
import sys
from pathlib import Path

import cadquery as cq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "packs" / "lab" / "assets" / "cad"


def revolve(profile):
    """Revolve an (r, z) profile given as a closed polyline around the Z axis."""
    return cq.Workplane("XZ").polyline(profile).close().revolve(360, (0, 0, 0), (0, 1, 0))


def microcentrifuge_tube_1_5ml():
    """1.5 ml tube: length 38.6, body 10.4, cap 11.4 x 7.0, conical tip 2.4 wide."""
    return revolve([
        (0, 0), (1.2, 0),        # flat tip, Ø 2.4
        (5.2, 8.0),              # 8.0 cone up to the body
        (5.2, 31.6),             # body Ø 10.4
        (5.7, 31.6), (5.7, 38.6),  # cap Ø 11.4, 7.0 high
        (0, 38.6),
    ])


def centrifuge_tube_50ml():
    """50 ml tube: length 115, body 28, cap 30 x 15, conical tip 10 wide."""
    return revolve([
        (0, 0), (5.0, 0),        # flat tip, Ø 10
        (14.0, 20.0),            # 20 cone up to the body
        (14.0, 100.0),           # body Ø 28
        (15.0, 100.0), (15.0, 115.0),  # cap Ø 30, 15 high
        (0, 115.0),
    ])


def pipette_tip_200ul():
    """200 µl tip: length 53, collar Ø 7.0, conical body to a 1.0 orifice, hollow (0.4 wall)."""
    outer = revolve([(0, 0), (0.5, 0), (3.5, 39.0), (3.5, 53.0), (0, 53.0)])
    inner = revolve([(0, -1.0), (0.1, -1.0), (3.1, 39.0), (3.1, 54.0), (0, 54.0)])
    return outer.cut(inner)


def vessel(pts, wall):
    """Open vessel from an outer (r, z) profile (bottom centre up to the rim), hollowed from the top."""
    top = pts[-1][1]
    solid = lib.revolve(list(pts) + [(0.0, top)])
    return solid.faces(">Z").shell(-wall)


def ring(pts, wall):
    """Tube open at both ends from an outer (r, z) profile; the wall is `wall` thick (radially)."""
    inner = [(max(r - wall, 0.2), z) for r, z in pts]
    return lib.revolve(list(pts) + inner[::-1])


def round_bottom(length, d, wall):
    """Round-bottom tube / test tube: hemispherical bottom, open at the top."""
    r = d / 2
    pts = [(r * math.sin(math.radians(a)), r - r * math.cos(math.radians(a))) for a in range(0, 91, 10)]
    pts.append((r, length))
    return vessel(pts, wall)


def measuring_cylinder(height, od, foot_d, wall=1.8):
    r, fr = od / 2, foot_d / 2
    return vessel([(0, 0), (fr, 0), (fr, 3.0), (r + 3.0, 12.0), (r, 16.0), (r, height)], wall)


def beaker(d, height, wall=2.0):
    return vessel([(0, 0), (d / 2, 0), (d / 2, height)], wall)


def erlenmeyer(d, height, neck_d, wall=1.6):
    """Narrow-neck Erlenmeyer flask (DIN 12380 shape): cylindrical lower body, cone, neck with lip."""
    R, rn = d / 2, neck_d / 2
    return vessel([(0, 0), (R - 5, 0), (R, 5.0), (R, 0.22 * height), (rn, 0.78 * height), (rn, height - 3.0),
                   (rn + 2.0, height - 3.0), (rn + 2.0, height)], wall)


def round_bottom_flask(d, height, neck_d, wall=1.6):
    """Round-bottom flask with a long neck (ground joint not modelled, plain neck of the joint's size)."""
    Rs, rn = d / 2, neck_d / 2
    th = 180.0 - math.degrees(math.asin(rn / Rs))
    pts = [(Rs * math.sin(math.radians(a)), Rs - Rs * math.cos(math.radians(a))) for a in [k * th / 24 for k in range(25)]]
    pts += [(rn, height - 3.0), (rn + 1.5, height - 3.0), (rn + 1.5, height)]
    return vessel(pts, wall)


def volumetric_flask(d, height, neck_d, shoulder=0.55, wall=1.6):
    R, rn = d / 2, neck_d / 2
    zs = height * shoulder
    return vessel([(0, 0), (R - 6, 0), (R, 6.0), (R, zs), (rn + 3, zs + 22.0), (rn, zs + 30.0), (rn, height)], wall)


def screw_vial(d, height):
    """Closed glass vial with screw cap (solid)."""
    r = d / 2
    return lib.revolve([(0, 0), (r - 1.0, 0), (r, 1.0), (r, height * 0.78), (r - 1.5, height * 0.84), (r - 1.5, height * 0.88),
                        (r - 0.5, height * 0.88), (r - 0.5, height), (0, height)])


def funnel(d, stem_d=10.0, stem_len=70.0, wall=1.2):
    r = d / 2
    cone_h = (r - stem_d / 2) / math.tan(math.radians(30))
    return ring([(stem_d / 2, 0), (stem_d / 2, stem_len), (r, stem_len + cone_h)], wall)


def burette(length=800.0, od=17.0, wall=1.4):
    body = ring([(2.5, 0), (2.5, 20), (od / 2, 50), (od / 2, length)], wall)
    stop = cq.Workplane("XZ").center(0, 45.0).circle(7.0).extrude(15.0, both=True)
    return body.union(stop)


def vol_pipette(length=335.0):
    return ring([(1.0, 0), (3.5, 20), (3.5, 140), (10.0, 155), (10.0, 200), (3.5, 215), (3.5, length)], 0.8)


def rod(length, d):
    return cq.Workplane("XY").circle(d / 2).extrude(length).faces("<Z or >Z").edges().fillet(d * 0.3)


def petri(d=90.0, h=15.6):
    base = lib.revolve([(0, 0), (43.5, 0), (43.5, 14.0), (42.5, 14.0), (42.5, 1.0), (0, 1.0)])
    lid = lib.revolve([(0, 15.6), (45.0, 15.6), (45.0, 0.6), (44.0, 0.6), (44.0, 14.6), (0, 14.6)])
    return base.add(lid).combine(glue=False) if False else cq.Workplane("XY").add(base.val()).add(lid.val()).combine()


def weighing_boat(side, h, wall=0.5):
    """Square weighing boat: sloped walls (bottom narrower), open top."""
    b = cq.Workplane("XY").rect(side * 0.7, side * 0.7).workplane(offset=h).rect(side, side).loft()
    return b.faces(">Z").shell(-wall)


def mtp(n, rows, pitch, a1x, a1y, well, depth, square=False):
    kw = {"well_sq": well} if square else {"well_d": well}
    return lambda: lib.plate_grid(127.76, 85.48, 14.4, n, rows, a1x, a1y, pitch, pitch, depth=depth, corner=3.0, **kw)


def box_grid(nx, pitch, well, depth=46.0):
    x0 = (133.0 - (nx - 1) * pitch) / 2
    return lambda: lib.plate_grid(133.0, 133.0, 50.0, nx, nx, x0, x0, pitch, pitch, well_d=well, depth=depth, corner=4.0)


def tip(length, top, orifice, cone):
    return lambda: lib.pipette_tip(length, top, orifice, wall=0.4, cone_len=cone)


# (builder, expected bounding box x, y, z in mm; compared sorted).
# Sources: SBS/ANSI SLAS 1-2004 footprint, ISO 4788 / ISO 3819 shapes, catalogue heights (see the report); tubes by typical maker sizes.
MORE = {
    "demo-lab-3005-microcentrifuge-tube-2ml": (lambda: lib.tube(41.0, 10.8, 2.4, 8.0, 11.4, 7.0), (11.4, 11.4, 41.0)),
    "demo-lab-3006-microcentrifuge-tube-0-5ml": (lambda: lib.tube(30.0, 7.9, 2.0, 7.0, 8.6, 5.0), (8.6, 8.6, 30.0)),
    "demo-lab-3007-centrifuge-tube-15ml": (lambda: lib.tube(120.0, 15.8, 4.0, 22.0, 17.0, 12.0), (17.0, 17.0, 120.0)),
    "demo-lab-3008-centrifuge-tube-50ml-sterile": (centrifuge_tube_50ml, (30.0, 30.0, 115.0)),
    "demo-lab-3009-cryo-tube-2ml": (lambda: lib.tube(48.0, 12.0, 10.0, 3.0, 12.5, 14.0), (12.5, 12.5, 48.0)),
    "demo-lab-3010-sample-tube-5ml-round-bottom": (lambda: round_bottom(75.0, 12.0, 0.8), (12.0, 12.0, 75.0)),
    "demo-lab-3022-screw-cap-tube-4ml": (lambda: lib.tube(55.0, 13.0, 6.0, 6.0, 14.0, 10.0), (14.0, 14.0, 55.0)),
    "demo-lab-2008-pipette-tip-10ul": (tip(46.0, 5.8, 0.6, 34.0), (5.8, 5.8, 46.0)),
    "demo-lab-2013-pipette-tip-10ul-sterile": (tip(46.0, 5.8, 0.6, 34.0), (5.8, 5.8, 46.0)),
    "demo-lab-2014-pipette-tip-2ul": (tip(40.0, 5.4, 0.5, 30.0), (5.4, 5.4, 40.0)),
    "demo-lab-2010-pipette-tip-20ul": (tip(53.0, 6.2, 0.7, 40.0), (6.2, 6.2, 53.0)),
    "demo-lab-2011-filter-pipette-tip-200ul": (tip(53.0, 7.0, 1.0, 39.0), (7.0, 7.0, 53.0)),
    "demo-lab-2012-pipette-tip-200ul-bag": (tip(53.0, 7.0, 1.0, 39.0), (7.0, 7.0, 53.0)),
    "demo-lab-2009-pipette-tip-1000ul": (tip(71.0, 9.8, 1.0, 55.0), (9.8, 9.8, 71.0)),
    "demo-lab-2015-filter-pipette-tip-1000ul": (tip(71.0, 9.8, 1.0, 55.0), (9.8, 9.8, 71.0)),
    "demo-lab-3011-microplate-96-wells": (mtp(12, 8, 9.0, 14.38, 11.24, 6.9, 10.8), (127.76, 85.48, 14.4)),
    "demo-lab-3012-microplate-384-wells": (mtp(24, 16, 4.5, 12.13, 8.99, 3.6, 11.5, square=True), (127.76, 85.48, 14.4)),
    "demo-lab-6001-test-tube-rack-24": (lambda: lib.tube_rack(140.0, 100.0, 70.0, 6, 4, 22.0, 18.0), (140.0, 100.0, 70.0)),
    "demo-lab-6002-rack-50ml-tubes-15": (lambda: lib.tube_rack(195.0, 125.0, 110.0, 5, 3, 36.0, 31.0), (195.0, 125.0, 110.0)),
    "demo-lab-6003-rack-1-5ml-tubes-96": (lambda: lib.tube_rack(192.0, 132.0, 45.0, 12, 8, 15.0, 11.2), (192.0, 132.0, 45.0)),
    "demo-lab-6013-rack-2ml-tubes-48": (lambda: lib.tube_rack(132.0, 102.0, 45.0, 8, 6, 15.0, 11.2), (132.0, 102.0, 45.0)),
    "demo-lab-6004-cryo-box-81": (box_grid(9, 13.9, 13.0), (133.0, 133.0, 50.0)),
    "demo-lab-6014-cryo-box-81-polycarbonate": (box_grid(9, 13.9, 13.0), (133.0, 133.0, 50.0)),
    "demo-lab-6005-sample-box-100": (box_grid(10, 12.9, 12.0), (133.0, 133.0, 50.0)),
    "demo-lab-6015-sample-archive-box-card-100": (box_grid(10, 12.9, 12.0), (133.0, 133.0, 50.0)),
    "demo-lab-6012-sample-box-25": (box_grid(5, 25.0, 22.0), (133.0, 133.0, 50.0)),
    "demo-lab-3003-petri-dish-90": (petri, (90.0, 90.0, 15.6)),
    "demo-lab-3013-weighing-boat-80x80": (lambda: weighing_boat(80.0, 14.0), (80.0, 80.0, 14.0)),
    "demo-lab-3018-weighing-boat-40x40": (lambda: weighing_boat(40.0, 9.0), (40.0, 40.0, 9.0)),
    "demo-lab-4001-measuring-cylinder-100ml": (lambda: measuring_cylinder(250.0, 31.0, 58.0), (58.0, 58.0, 250.0)),
    "demo-lab-4002-measuring-cylinder-250ml": (lambda: measuring_cylinder(335.0, 42.0, 72.0), (72.0, 72.0, 335.0)),
    "demo-lab-4003-measuring-cylinder-1000ml": (lambda: measuring_cylinder(465.0, 68.0, 110.0), (110.0, 110.0, 465.0)),
    "demo-lab-4020-measuring-cylinder-10ml": (lambda: measuring_cylinder(140.0, 16.0, 38.0, 1.5), (38.0, 38.0, 140.0)),
    "demo-lab-4004-erlenmeyer-flask-250ml": (lambda: erlenmeyer(85.0, 140.0, 34.0), (85.0, 85.0, 140.0)),
    "demo-lab-4005-erlenmeyer-flask-500ml": (lambda: erlenmeyer(105.0, 175.0, 34.0), (105.0, 105.0, 175.0)),
    "demo-lab-4018-erlenmeyer-flask-100ml": (lambda: erlenmeyer(64.0, 105.0, 28.0), (64.0, 64.0, 105.0)),
    "demo-lab-4006-round-bottom-flask-500ml-ns29-32": (lambda: round_bottom_flask(105.0, 163.0, 34.0), (105.0, 105.0, 163.0)),
    "demo-lab-4019-round-bottom-flask-250ml-ns29-32": (lambda: round_bottom_flask(85.0, 140.0, 34.0), (85.0, 85.0, 140.0)),
    "demo-lab-4007-beaker-250ml-low-form": (lambda: beaker(70.0, 95.0), (70.0, 70.0, 95.0)),
    "demo-lab-4008-beaker-1000ml-tall-form": (lambda: beaker(95.0, 180.0), (95.0, 95.0, 180.0)),
    "demo-lab-4009-test-tube-16x160": (lambda: round_bottom(160.0, 16.0, 1.0), (16.0, 16.0, 160.0)),
    "demo-lab-4010-volumetric-flask-100ml": (lambda: volumetric_flask(60.0, 170.0, 19.0), (60.0, 60.0, 170.0)),
    "demo-lab-4011-volumetric-pipette-10ml": (vol_pipette, (20.0, 20.0, 335.0)),
    "demo-lab-4012-burette-50ml": (burette, (17.0, 30.0, 800.0)),
    "demo-lab-4016-glass-funnel-75mm": (lambda: funnel(75.0), (75.0, 75.0, 70.0 + 32.5 / math.tan(math.radians(30)))),
    "demo-lab-4017-glass-stirring-rod-200x6": (lambda: rod(200.0, 6.0), (6.0, 6.0, 200.0)),
    "demo-lab-4021-screw-cap-vial-20ml": (lambda: screw_vial(25.0, 65.0), (25.0, 25.0, 65.0)),
}

PARTS = {
    "demo-lab-3001-microcentrifuge-tube-1-5ml": microcentrifuge_tube_1_5ml,
    "demo-lab-3002-centrifuge-tube-50ml": centrifuge_tube_50ml,
    "demo-lab-2002-pipette-tip-200ul": pipette_tip_200ul,
}
EXPECT = {k: v[1] for k, v in MORE.items()}
EXPECT.update({"demo-lab-3001-microcentrifuge-tube-1-5ml": (11.4, 11.4, 38.6), "demo-lab-3002-centrifuge-tube-50ml": (30.0, 30.0, 115.0),
               "demo-lab-2002-pipette-tip-200ul": (7.0, 7.0, 53.0)})
PARTS.update({k: v[0] for k, v in MORE.items()})

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, build in PARTS.items():
        path = OUT / f"{name}.step"
        lib.export_step(build(), path)
        bb = build().val().BoundingBox()
        print(f"  {path.name}  x {bb.xlen:.2f}  y {bb.ylen:.2f}  z {bb.zlen:.2f}")

