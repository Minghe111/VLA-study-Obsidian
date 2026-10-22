"""Lab pack, series 4: glassware. Measuring cylinders, Erlenmeyer and round-bottom
flasks, beakers, test tubes, a volumetric flask and pipette, a burette, a separating
funnel, a desiccator, a condenser, a glass funnel, stirring rods and screw-cap vials.

Glass is 'glass' with alpha 0.5. Graduation marks are white. Caps are blue. The
stopcocks and clamps are plain. No lettering. Units mm, product stands on z=0,
front faces -Y.
"""

GLASS = 'glass'
WHITE = '#F7F7F5'
CAP_BLUE = '#2F6FE0'


def _glass(k, r, h, z, x=0, y=0, **kw):
    """Clear borosilicate tube section centred at (x, y, z) (alpha 0.5, slight gloss)."""
    return k.cyl(r, h, (x, y, z), GLASS, alpha=0.5, rough=0.1, **kw)


def stand(k, z_arm, ring_r):
    """Black lab stand: base plate, a rod at the back and a ring clamp at z_arm."""
    k.box((200, 160, 12), (0, 0, 6), 'dark', bevel=2, metal=0.5, rough=0.4)               # base plate
    k.cyl(6, z_arm - 12, (0, 70, (12 + z_arm) / 2), 'metal', metal=0.8, rough=0.3)        # rod
    arm = 70 - (ring_r + 4)
    k.cyl(4, arm, (0, (70 + ring_r + 4) / 2, z_arm), 'metal', rot=(90, 0, 0), metal=0.8, rough=0.3)
    k.torus(ring_r, 4, (0, 0, z_arm), 'dark', metal=0.5, rough=0.4)                       # ring clamp


def measuring_cylinder(k, d, h, base_d, step, base_h=4):
    """Graduated measuring cylinder: hexagonal glass foot, straight tube, white marks, open top."""
    r = d / 2
    k.cyl(base_d / 2, base_h, (0, 0, base_h / 2), GLASS, alpha=0.5, rough=0.1, verts=6)  # hexagonal foot
    _glass(k, r, h, base_h + h / 2)                                                       # tube
    _glass(k, r + 0.8, 1.6, base_h + h)                                                   # rim
    for z in range(step, int(h), step):
        k.cyl(r + 0.4, 0.8, (0, 0, base_h + z), WHITE, verts=40)                          # graduation mark


def erlenmeyer(k, body_d, body_h, neck_d, neck_h):
    """Conical flask: flat wide base, sloping sides, narrow neck with a plain rim."""
    _glass(k, body_d / 2, body_h, body_h / 2, r2=neck_d / 2)                              # conical body
    _glass(k, neck_d / 2, neck_h, body_h + neck_h / 2)                                    # neck
    _glass(k, neck_d / 2 + 2, 3, body_h + neck_h + 1.5)                                   # rim


def round_bottom(k, sr, neck_d, neck_h, joint_d, joint_h):
    """Round-bottom flask in a ring clamp on a stand: glass sphere, short neck, ground joint."""
    hr = 0.6 * sr                                   # ring height
    ring_r = 0.72 * sr                              # ring radius = sphere radius at the ring
    zc = hr + 0.694 * sr                            # sphere centre so the ring meets the equator
    k.sphere(sr, (0, 0, zc), GLASS, alpha=0.5, rough=0.1)
    top = zc + sr - 4
    _glass(k, neck_d / 2, neck_h, top + neck_h / 2)                                       # neck
    k.cyl(neck_d / 2, joint_h, (0, 0, top + neck_h + joint_h / 2), GLASS, r2=joint_d / 2,
          alpha=0.5, rough=0.1)                                                           # ground joint
    stand(k, hr, ring_r)


def beaker(k, d, h, step):
    """Low or tall beaker: clear straight wall, flat bottom, a spout at the rim, white marks."""
    r = d / 2
    _glass(k, r, 3, 1.5)                                                                  # bottom
    _glass(k, r, h, h / 2)                                                                # wall
    k.box((8, 10, 3), (r + 1.5, 0, h), GLASS, alpha=0.5, rough=0.1)                       # spout lip
    for z in range(step, int(h), step):
        k.cyl(r + 0.4, 0.8, (0, 0, z), WHITE, verts=40)                                   # graduation


def test_tubes(k):
    """Ten test tubes in two rows of five, standing on the floor, open rims, rounded bottoms."""
    for i in range(5):
        for j in range(2):
            x = (i - 2) * 22
            y = (j - 0.5) * 22
            k.sphere(8, (x, y, 8), GLASS, alpha=0.5, rough=0.1)
            _glass(k, 8, 152, 8 + 76, x, y)
            _glass(k, 9, 3, 158.5, x, y)


def volumetric_flask(k):
    """Volumetric flask: round pear-shaped bulb with a flat bottom, a long narrow neck with one thin ring mark and a stopper."""
    _glass(k, 32, 40, 20)                                                                 # flat-bottom round body
    k.sphere(32, (0, 0, 40), GLASS, scale=(1, 1, 0.9), alpha=0.5, rough=0.1)             # rounded top of the bulb
    _glass(k, 6.5, 110, 62 + 55)                                                          # long narrow neck, d 13
    k.cyl(7.0, 0.8, (0, 0, 150), WHITE, verts=40)                                         # one thin ring mark
    k.cyl(8, 18, (0, 0, 183), 'white', bevel=1, verts=40)                                 # plastic stopper


def pipette(k):
    """Class A volumetric pipette: slender glass stem, large bulb, tip, one mark on top."""
    k.cyl(1.2, 40, (0, 0, 20), GLASS, r2=3, alpha=0.5, rough=0.1)                        # tip, narrow at the bottom
    _glass(k, 3, 165, 112.5)                                                              # lower stem, z 30..195
    k.sphere(15, (0, 0, 200), GLASS, scale=(1, 1, 1.4), alpha=0.5, rough=0.1)             # bulb
    _glass(k, 3.5, 135, 287.5)                                                            # upper stem, z 220..355
    _glass(k, 4, 4, 357)                                                                  # top rim
    k.cyl(3.6, 0.6, (0, 0, 300), WHITE, verts=32)                                         # mark


def burette(k):
    """50 ml burette on a stand: long tube, white marks, narrow tip, glass stopcock near the tip."""
    A = k.ACCENT['lab']
    k.box((220, 160, 12), (0, 0, 6), 'dark', bevel=2, metal=0.5, rough=0.4)               # base plate
    k.cyl(6, 520, (0, 60, 272), 'metal', metal=0.8, rough=0.3)                            # stand rod, top just above the tube rim
    k.cyl(4, 52, (0, 34, 300), 'metal', rot=(90, 0, 0), metal=0.8, rough=0.3)             # clamp arm
    k.box((22, 16, 24), (0, 12, 300), 'dark', bevel=2)                                    # clamp jaw
    k.box((22, 3, 8), (0, 3.5, 300), A)                                                   # accent
    _glass(k, 7, 420, 290)                                                                # tube
    _glass(k, 9, 4, 502)                                                                  # top rim
    _glass(k, 1.2, 20, 70, r2=7)                                                          # tip cone
    _glass(k, 1.2, 12, 54)                                                                # nozzle
    for z in range(120, 500, 50):
        k.cyl(7.15, 0.8, (0, 0, z), WHITE, verts=48)                                      # graduation
    k.cyl(3.5, 26, (0, 0, 72), GLASS, rot=(0, 90, 0), alpha=0.5, rough=0.1)               # stopcock
    k.box((24, 4, 5), (0, 0, 79), 'light', bevel=1)                                       # handle


def separating_funnel(k):
    """500 ml separating funnel: pear body, narrow stoppered neck, stem, white stopcock, ring clamp."""
    _glass(k, 12, 26, 37)                                                                 # stem, z 24..50
    k.cyl(12, 90, (0, 0, 95), GLASS, r2=50, alpha=0.5, rough=0.1)                        # lower cone
    _glass(k, 50, 50, 165)                                                                # wide body
    k.cyl(50, 42, (0, 0, 211), GLASS, r2=20, alpha=0.5, rough=0.1)                       # upper cone
    _glass(k, 20, 10, 237)                                                                # neck
    _glass(k, 22, 4, 243)                                                                 # neck rim
    _glass(k, 16, 14, 240)                                                                # stopper
    _glass(k, 20, 6, 250)                                                                 # stopper knob
    _glass(k, 3.5, 24, 12)                                                                # tip
    k.cyl(7, 40, (0, 0, 20), 'white', rot=(0, 90, 0), bevel=1)                            # PTFE stopcock
    k.box((24, 6, 6), (0, 0, 31), 'white', bevel=1)                                       # stopcock handle
    stand(k, 100, 40)


def desiccator(k):
    """250 mm desiccator (inside 250, about 324 tall with lid): a tall thick-walled glass vessel
    narrowing slightly at the bottom, a wide ground flange at the top, a domed glass lid with a
    knob, and a white perforated porcelain plate at the waist."""
    _glass(k, 118, 120, 60)                                                               # lower body
    _glass(k, 128, 110, 175)                                                              # upper body, wider
    _glass(k, 140, 10, 235)                                                               # wide ground flange
    k.cyl(16, 116, (0, 0, 64), 'white', verts=32)                                         # central porcelain pedestal under the plate
    k.cyl(122, 5, (0, 0, 122), 'white', verts=64)                                         # perforated porcelain plate
    for i in range(-3, 4):
        for j in range(-3, 4):
            if i * i + j * j <= 9:
                k.cyl(5, 5.4, (i * 28, j * 28, 122), '#9AA0A8')                           # its holes
    k.sphere(140, (0, 0, 240), GLASS, scale=(1, 1, 0.42), alpha=0.5, rough=0.1)          # domed glass lid
    k.cyl(14, 26, (0, 0, 312), GLASS, alpha=0.5, rough=0.1)                              # lid knob


def condenser(k):
    """300 mm Liebig condenser clamped vertically to a stand: outer water jacket with two side
    hose nipples, a long straight inner tube, a ground joint at each end."""
    k.cyl(20, 300, (0, 0, 190), GLASS, alpha=0.5, rough=0.1)                             # jacket, z 40..340
    k.cyl(7.5, 340, (0, 0, 190), GLASS, alpha=0.5, rough=0.1)                            # inner tube, z 20..360
    for z in (40, 340):
        k.cyl(12, 20, (0, 0, z), GLASS, alpha=0.5, rough=0.1)                            # ground joint at each end
    for z in (70, 310):
        k.cyl(6, 26, (33, 0, z), GLASS, rot=(0, 90, 0), alpha=0.5, rough=0.1)            # hose nipple, pointing sideways
    stand(k, 190, 24)                                                                    # clamp round the jacket


def glass_funnel(k):
    """75 mm glass funnel: a clear cone with a smooth rim, standing on its short stem."""
    _glass(k, 3.5, 24, 12)                                                                # stem
    k.cyl(3.5, 60, (0, 0, 54), GLASS, r2=38, alpha=0.5, rough=0.1)                       # cone
    _glass(k, 38.5, 2.5, 85.25)                                                           # rim


def stirring_rods(k):
    """Ten solid glass rods, 6 mm x 200 mm, lying parallel on the floor in a flat bundle."""
    for i in range(10):
        k.cyl(3, 200, (0, (i - 4.5) * 12, 3), GLASS, rot=(0, 90, 0), alpha=0.5, rough=0.1, verts=24)


def vials(k, cols, rows, pitch):
    """Screw-cap glass vials standing in a grid, each with a blue cap."""
    for i in range(cols):
        for j in range(rows):
            x = (i - (cols - 1) / 2) * pitch
            y = (j - (rows - 1) / 2) * pitch
            _glass(k, 12.5, 54, 27, x, y)
            k.cyl(13.5, 14, (x, y, 51), CAP_BLUE, bevel=1, verts=40)


SKUS = {
    'DEMO-LAB-4001': (measuring_cylinder, {'d': 38, 'h': 200, 'base_d': 60, 'step': 15}),
    'DEMO-LAB-4002': (measuring_cylinder, {'d': 50, 'h': 270, 'base_d': 80, 'step': 20, 'base_h': 6}),
    'DEMO-LAB-4003': (measuring_cylinder, {'d': 75, 'h': 380, 'base_d': 110, 'step': 40, 'base_h': 6}),
    'DEMO-LAB-4004': (erlenmeyer, {'body_d': 90, 'body_h': 95, 'neck_d': 30, 'neck_h': 40}),
    'DEMO-LAB-4005': (erlenmeyer, {'body_d': 110, 'body_h': 115, 'neck_d': 36, 'neck_h': 45}),
    'DEMO-LAB-4006': (round_bottom, {'sr': 50, 'neck_d': 26, 'neck_h': 40, 'joint_d': 32, 'joint_h': 30}),
    'DEMO-LAB-4007': (beaker, {'d': 70, 'h': 85, 'step': 25}),
    'DEMO-LAB-4008': (beaker, {'d': 105, 'h': 150, 'step': 40}),
    'DEMO-LAB-4009': (test_tubes, {}, {'pitch': 40, 'yaw': -25}),
    'DEMO-LAB-4010': (volumetric_flask, {}),
    'DEMO-LAB-4011': (pipette, {}, {'fill': 0.85}),
    'DEMO-LAB-4012': (burette, {}, {'fill': 0.85}),
    'DEMO-LAB-4013': (separating_funnel, {}, {'fill': 0.85}),
    'DEMO-LAB-4014': (desiccator, {}),
    'DEMO-LAB-4015': (condenser, {}),
    'DEMO-LAB-4016': (glass_funnel, {}),
    'DEMO-LAB-4017': (stirring_rods, {}, {'fill': 0.7}),
    'DEMO-LAB-4018': (erlenmeyer, {'body_d': 65, 'body_h': 66, 'neck_d': 24, 'neck_h': 28}),
    'DEMO-LAB-4019': (round_bottom, {'sr': 36, 'neck_d': 20, 'neck_h': 34, 'joint_d': 25, 'joint_h': 24}),
    'DEMO-LAB-4020': (measuring_cylinder, {'d': 17, 'h': 125, 'base_d': 30, 'step': 8, 'base_h': 3}, {'fill': 0.85}),
    'DEMO-LAB-4021': (vials, {'cols': 4, 'rows': 3, 'pitch': 36}),
}

HINTS = {
    'DEMO-LAB-4001': 'A 100 ml measuring cylinder of clear borosilicate glass: a straight tube on a hexagonal foot, with white graduation lines only, no numerals, along the tube and an open top rim. Standing alone, upright. The clear glass stands out clearly against the white background. No text, no numbers, no logos, no labels anywhere on the product.',
    'DEMO-LAB-4002': 'A 250 ml measuring cylinder of clear borosilicate glass: a straight tube on a wide hexagonal foot, with white graduation lines only, no numerals, along the tube and an open top rim. Standing alone, upright. The clear glass stands out clearly against the white background. No text, no numbers, no logos, no labels anywhere on the product.',
    'DEMO-LAB-4003': 'A 1000 ml measuring cylinder of clear borosilicate glass: a tall, wide straight tube on a broad hexagonal foot, with widely spaced white graduation lines only, no numerals, and an open top rim. Standing alone, upright. The clear glass stands out clearly against the white background. No text, no numbers, no logos, no labels anywhere on the product.',
    'DEMO-LAB-4004': 'A 250 ml Erlenmeyer flask of clear borosilicate glass: a conical flask with a flat wide base, sloping straight sides and a narrow cylindrical neck with a plain rim, standing upright alone. Plain clear glass with no printed panel, no scale and no lettering. The clear glass stands out clearly against the white background. No text, no numbers, no logos, no labels anywhere on the product.',
    'DEMO-LAB-4005': 'A 500 ml Erlenmeyer flask of clear borosilicate glass: a large conical flask with a flat wide base, sloping straight sides and a short narrow neck with a plain rim, standing upright alone. The clear glass stands out clearly from the white background with a soft shadow. Not a round-bottom flask: the base is flat.',
    'DEMO-LAB-4006': 'A 500 ml round-bottom flask of clear borosilicate glass: a perfect glass sphere with a short narrow neck ending in a ground glass joint, resting upright in a black cork ring on the bench. No clamp, no handle, no spout, nothing touching the neck. The clear glass stands out clearly against the white background. No text, no numbers, no logos, no labels anywhere on the product.',
    'DEMO-LAB-4007': 'A 250 ml low-form beaker of clear borosilicate glass: a short, wide, straight-walled cylinder with a flat bottom and one small rounded pouring lip at the rim, standing upright alone. Nothing else attached to the rim, no tab, no handle. White graduation lines only on the side, no numerals. The clear glass stands out clearly against the white background. No text, no numbers, no logos, no labels anywhere on the product.',
    'DEMO-LAB-4008': 'A 1000 ml tall-form beaker of clear borosilicate glass: a tall, straight-walled cylinder, clearly taller than it is wide (about 1.5 times its width), with a flat bottom and one small pouring lip at the rim, standing upright alone. White graduation lines only on the side, no numerals. Nothing attached to the rim. The clear glass stands out clearly against the white background. No text, no numbers, no logos, no labels anywhere on the product.',
    'DEMO-LAB-4009': 'Ten separate clear borosilicate test tubes, 16 by 160 mm, with plain open rims and rounded closed bottoms, standing upright in two rows of five. Each tube stands apart with gaps between them; no tube overlaps or passes into another. Exactly ten tubes, no packaging. The clear glass stands out clearly against the white background. No text, no numbers, no logos, no labels anywhere on the product.',
    'DEMO-LAB-4010': 'A 100 ml volumetric flask of clear borosilicate glass: a round pear-shaped bulb with a flat bottom and a rounded top, a long narrow neck with one thin white ring mark, and a plastic stopper on top. The bulb is plain glass with no lettering, no printed marks and no scale. Not a conical flask. The clear glass stands out clearly against the white background. No text, no numbers, no logos, no labels anywhere on the product.',
    'DEMO-LAB-4011': 'A 10 ml volumetric pipette of clear borosilicate glass, standing upright alone: a long slender stem about 360 mm tall, with a swollen round bulb at mid-length (halfway up the stem, not at the bottom), a narrow open delivery tip at the bottom, and one thin white calibration ring on the upper stem. Not a dropper, not a plastic pipette, not a test tube. No hooks, no bent parts, no stray parts. White marks only, no numerals. The clear glass stands out clearly from the white background with a soft shadow. No text, no numbers, no logos, no labels anywhere on the product.',
    'DEMO-LAB-4012': 'A 50 ml laboratory burette: a long thin straight clear glass tube, open at the top with no funnel, with fine white graduation marks only (no numerals), a glass stopcock with a small handle at the bottom and a narrow tip below it. The tube is held upright in a burette clamp on a black laboratory stand with a heavy rectangular base plate and one vertical steel rod. The rod ends just above the clamp; nothing floats above the rod or beside it. Not a thermometer. No text, no numbers, no logos, no labels anywhere on the product.',
    'DEMO-LAB-4013': 'A 500 ml separating funnel of clear borosilicate glass: a pear-shaped glass body with a glass stopper in the open top, a short glass stem below the body, and a white plastic stopcock with a short handle fixed at the very bottom tip of the stem, attached to it. Held in a black ring clamp round the body on a black laboratory stand with a heavy base plate and a vertical steel rod. No loose parts, nothing floating beside the stand. Not a vase, not a lamp. No text, no numbers, no logos, no labels anywhere on the product.',
    'DEMO-LAB-4014': 'A 250 mm desiccator of clear borosilicate glass: a tall thick-walled glass vessel with a wide flange at the top, a domed glass lid with a small glass knob on top, and inside, a white perforated porcelain plate resting on a central porcelain pedestal in the lower part, so the plate is supported from below and not floating. No labels or printing. No text, no numbers, no logos, no labels anywhere on the product.',
    'DEMO-LAB-4015': 'A 300 mm Liebig condenser of clear borosilicate glass, clamped vertically to a black laboratory stand: a wide outer water jacket with two small glass hose nipples pointing sideways, a long straight inner glass tube running through the jacket, and a ground glass joint at each end. A black clamp grips the middle of the jacket on a vertical steel rod. Nothing under the condenser: no base, no feet, no cradle. Not a horizontal tube, not a coil. No text, no numbers, no logos, no labels anywhere on the product.',
    'DEMO-LAB-4016': 'A 75 mm glass funnel of clear borosilicate glass: a clear cone with a wide smooth rim at the top and a short straight stem at the bottom, standing upright on its stem alone. The clear glass stands out clearly from the white background with a soft shadow. Not a separating funnel: it has no stopcock and no stopper.',
    'DEMO-LAB-4017': 'Ten solid clear glass stirring rods, exactly ten, each about 200 mm long and 6 mm thick with rounded ends, lying parallel and side by side with small gaps so that all ten are countable, in one flat row across the frame. Solid glass rods, not hollow tubes, not tubing. The clear glass stands out clearly from the white background with a soft shadow. No text, no numbers, no logos, no labels anywhere on the product.',
    'DEMO-LAB-4018': 'A 100 ml Erlenmeyer flask of clear borosilicate glass: a small conical flask with a flat wide base, sloping straight sides and a short narrow neck with a plain rim, standing upright alone. The clear glass stands out clearly from the white background with a soft shadow. Not a round-bottom flask: the base is flat.',
    'DEMO-LAB-4019': 'A 250 ml round-bottom flask of clear borosilicate glass: a glass sphere with a short narrow neck ending in a ground glass joint, resting upright in a black cork ring, the flask large in the frame. No handle. Clear glass standing out from the white background with a soft shadow.',
    'DEMO-LAB-4020': 'A 10 ml graduated measuring cylinder of clear borosilicate glass: a slim straight tube with an open top rim on a small hexagonal foot, with fine white graduation lines only (no numerals, no digits, no printed scale values) along the tube, standing upright alone. Not a pipette, not a test tube, not a beaker. The clear glass stands out clearly from the white background with a soft shadow. No text, no numbers, no logos, no labels anywhere on the product.',
    'DEMO-LAB-4021': 'Twelve clear borosilicate glass screw-cap vials of 20 ml, short and wide with flat bottoms, standing upright in a grid of four by three on the floor, each with a blue plastic screw cap. The clear glass and blue caps stand out clearly from the white background with a soft shadow. Not test tubes: the vials are short and wide and have caps.',
}

