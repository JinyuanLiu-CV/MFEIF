import math


def draw_viz(viz, y, x, title: str, rebuild: bool = False):
    if viz is None:
        return
    if math.isnan(y):
        raise Exception(title, "crash: invalid number")
    viz.line([y], [x], win=title, opts=dict(title=title), update='replace' if rebuild else 'append')
