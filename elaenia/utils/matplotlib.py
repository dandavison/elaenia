def float_to_rgb(f, colormap):
    r, g, b, a = [int(round(255 * p)) for p in colormap(f)]
    return "#" + "".join("%x" % n for n in [r, g, b])
