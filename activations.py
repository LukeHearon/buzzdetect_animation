import pandas as pd
from numpy import array
from manim import *

# Read activations data
activation_df = pd.read_csv('./data/morebees_buzzdetect.csv')
activation_df = activation_df.head(3)
frames_to_animate = 4

framelength = 0.96
framehop = 1
steptime = framelength * framehop

neurons = ['ins_buzz', 'ambient_noise', 'mech_auto']

cp = [BLUE, GREEN, YELLOW, RED]  # this is dumb


class ActivationDot(Dot):
    def __init__(self, activation, neuron, **kwargs):
        super().__init__(**kwargs)
        self.activation = activation
        self.neuron = neuron


class Activation(Scene):
    def make_mobs(self, neuron, mobcolor):
        activation = activation_df[neuron][0]

        dot = ActivationDot(
            radius=0.1,
            color=mobcolor,
            activation=activation,
            neuron=neuron,
            point=array([0, activation, 0])
        )
        
        path = TracedPath(
            dot.get_center,
            dissipating_time=0.5,
            stroke_opacity=[0, 1],
            color=mobcolor
        )

        return {'dot': dot, 'path': path}

    def construct(self):
        activation_sub = activation_df[neurons]

        plane = NumberPlane().add_coordinates()
        self.add(plane)

        mobs = []
        for i, neuron in enumerate(neurons):
            mob_dict = self.make_mobs(neuron, cp[i])
            mob_dict['neuron'] = neuron
            mobs.append(mob_dict)

        for mob_dict in mobs:
            self.add(mob_dict['dot'])
            self.add(mob_dict['path'])

        # Move the dot with independent X and Y animations
        for index, row in activation_sub.iterrows():
            x_target = steptime * (index + 1)

            def update_dot(dot, alpha):
                # Linear movement for X
                x_remainder = x_target - dot.get_x()
                x_update = alpha * x_remainder

                # Smoothed movement for Y
                y_remainder = dot.activation - dot.get_y()
                y_update = alpha * y_remainder

                dot.shift(array([x_update, y_update, 0]))

            updates = []
            for mob_dict in mobs:
                mob_dict['dot'].activation = row[mob_dict['neuron']]
                updates.append(UpdateFromAlphaFunc(mob_dict['dot'], update_dot))

            # Animate the dot
            self.play(
                updates,
                run_time=steptime,
                rate_func=linear,
            )

        self.wait()
