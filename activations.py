import pandas as pd
from numpy import array
from scipy.interpolate import CubicSpline
from manim import *

# TODO:
    # is it worth it to precalculate frames? Interpolation is prolly getting translated to C each call; could do it in one go


runrate = 1  # rate of playback as multiple of realtime; unfortunately affects interpolation
framelength = 0.96
framehop = 1
framehop_s = framelength * framehop

threshold = -1

labels = True

x_offset = framelength/2  # where should the dots land relative to frame end?

# Read activations data
activation_df = pd.read_csv('./data/morebees_buzzdetect.csv')
activation_df = activation_df.head(40)

neurons = ['ins_buzz', 'ambient_background', 'mech_auto']

cp = [BLUE, GREEN, YELLOW, RED, PURPLE]  # this is dumb


class ActivationDot(LabeledDot):
    def __init__(self, neuron, label='', **kwargs):
        self.neuron = neuron

        self.activations = activation_df[self.neuron]
        activation_0 = self.activations[0]
        self.interpolator = CubicSpline(activation_df['start'], self.activations)

        super().__init__(point=array([-x_offset, activation_0, 0]), label=label, **kwargs)


def make_mobs(neuron, mobcolor):
    label = ''
    if labels:
        label = Text(neuron, color=mobcolor, font_size=24)

    dot = ActivationDot(
        radius=0.1,
        color=mobcolor,
        neuron=neuron,
        label=label
    )

    moblist = [dot]  # must ALWAYS be 0th element

    path_long = TracedPath(
        dot.get_center,
        stroke_opacity=0.5,
        stroke_width=6,
        stroke_color=mobcolor
    )
    moblist.append(path_long)

    path_short = TracedPath(
        dot.get_center,
        dissipating_time=0.96,
        stroke_opacity=1,
        stroke_width=6,
        stroke_color=mobcolor
    )

    moblist.append(path_short)

    return moblist

class Activation(MovingCameraScene):
    def construct(self):
        y_width_original =  (4 + 1/9) * 2
        y_range = (-4, 8)  # leaving because I might assign it programmatically in the future
        y_width_new = y_range[1] - y_range[0]
        y_scale = y_width_new/y_width_original

        x_min = min(activation_df['start'])
        if x_min > 0:
            x_min = x_min - (8*y_scale)  # if we're not starting at 0s, expand to edge of frame

        x_max = max(activation_df['start']) + (8*y_scale)  # finish at edge of frame


        # Basic geometry
        #
        plane = NumberPlane(
            y_range=y_range,
            x_range=(x_min, x_max)
        )

        plane.add_coordinates()
        plane.align_to(self.camera.frame.get_center(), LEFT)
        # plane.align_to(self.camera.frame, DOWN)

        self.add(plane)

        thresholdline = Line(
            start=array([x_min, threshold, 0]),
            end=array([x_max, threshold, 0]),
            stroke_width=4,
            stroke_color=WHITE
        )

        self.add(thresholdline)

        mobs = []
        for i, neuron in enumerate(neurons):
            moblist = make_mobs(neuron, cp[i])
            mobs.append(moblist)

        for moblist in mobs:
            for mob in moblist:
                self.add(mob)

        self.play(self.camera.frame.animate.scale(y_scale), run_time=0.01)


        #  Animate
        #
        for t in activation_df['start']:
            x_target = t - x_offset

            def update_dot(dot, alpha):
                # Linear movement for X
                x_current = dot.get_x()
                x_remainder = x_target - x_current
                x_next = x_current + (alpha * x_remainder)

                # Smoothed movement for Y
                y_next = dot.interpolator(x_next)

                dot.move_to(array([x_next, y_next, 0]))

            updates = []
            for moblist in mobs:
                updates.append(UpdateFromAlphaFunc(moblist[0], update_dot))

            # Animate the dot
            self.play(
                updates,
                self.camera.frame.animate.shift(RIGHT * framehop_s),
                run_time=framehop_s/runrate,
                rate_func=linear
            )

        self.wait()
