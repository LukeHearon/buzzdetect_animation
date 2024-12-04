import pandas as pd
from numpy import array
from scipy.interpolate import CubicSpline
from manim import *

# TODO:
# change color based on active or not
# have special animation for out of bounds (and adjust y axis to fit most neuron values)

# function inputs
#
activation_df = pd.read_csv('./data/morebees_buzzdetect.csv')
neurons = ['ins_buzz', 'ambient_background', 'human']
framelength = 0.96
threshold = -1
labels = True


# set up activation data
#

# center activations on frame
activation_df['mid'] = activation_df['start'] + (framelength/2)

# make dots start at 0
activation_df.index = activation_df.index + 1
row_start = {'mid': min(activation_df['mid'] - framelength/2)}

for n in neurons:
    row_start.update({n: 0})

row_start = pd.DataFrame(row_start, index = [0])
activation_df = pd.concat([row_start, activation_df])

# subset
activation_df = activation_df[['mid'] + neurons]


cp = [BLUE, GREEN, YELLOW, RED, PURPLE]  # this is dumb


class ActivationDot(LabeledDot):
    def __init__(self, neuron, label='', **kwargs):
        self.neuron = neuron
        self.interpolator = CubicSpline(activation_df['mid'], activation_df[neuron], extrapolate=True)

        super().__init__(label=label, **kwargs)


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

    path = TracedPath(
        dot.get_center,
        stroke_opacity=1,
        stroke_width=6,
        stroke_color=mobcolor
    )

    return [dot, path]


class Activations(MovingCameraScene):
    def construct(self):
        # Make axes
        #
        y_width_original = (4 + 1 / 9) * 2
        
        y_min = max(activation_df[neurons].min().min(), -4) - 0.25
        y_max = min(activation_df[neurons].max().max(), 7) + 0.25

        y_width_new = y_max - y_min
        y_scale = y_width_new / y_width_original

        time_start = min(activation_df['mid'])
        time_end = max(activation_df['mid'])
        time_span = time_end - time_start

        x_min = time_start
        if x_min > 0:
            x_min = x_min - (8 * y_scale)  # if we're not starting at 0s, expand to edge of frame

        x_max = time_end + (8 * y_scale)  # finish at edge of frame

        axes = Axes(
            x_range=[x_min, x_max, 1],
            y_range=[y_min, y_max, 1],
            tips=False,
            x_length= (x_max - x_min) * 1.5,
            axis_config={"include_numbers": True}
        )

        axes.scale_to_fit_height(self.camera.frame.height)

        # oddly, the y axis starts just a smidge too far left, then corrects itself over time
        # not a big enough issue to correct right now...
        # TODO: correct
        axes.align_to(self.camera.frame.get_center(), LEFT)
        self.add(axes)

        def update_dot(mobj):
            x_point = axes.y_axis.get_right()[0] - axes.y_axis.tick_size

            # NOTE: we might get weird extrapolation since the dots have to start _somewhere_
            x_coord = axes.p2c(array([x_point, 0, 0]))[0]
            y_coord = mobj.interpolator(x_coord)

            y_point = axes.c2p(0, y_coord)[1]

            mobj.move_to(array([x_point, y_point, 0]))

        mobs = [make_mobs(neuron, cp[i]) for i, neuron in enumerate(neurons)]

        for moblist in mobs:
            moblist[0].add_updater(update_dot, call_updater=True)
            self.add(*moblist)

        scene_movement = axes.c2p(time_end, 0) - axes.c2p(time_start, 0)
        self.play(
            self.camera.frame.animate.shift(scene_movement),
            axes.y_axis.animate.shift(scene_movement),
            rate_func=rate_functions.linear,
            run_time=time_span
        )

