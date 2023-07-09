from random import random
from typing import Callable

from kivy.graphics import Color, Line, Ellipse
from kivy.uix.widget import Widget


class PaintWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._line = Line()

    def on_touch_down(self, touch):
        color = (random(), random(), random())
        with self.canvas:
            Color(*color)
            width = 10
            Ellipse(size=(width * 2, width * 2), pos=(touch.x - width, touch.y - width))
            touch.ud['line'] = Line(width=width)
            self.on_touch_move(touch)

    def on_touch_move(self, touch):
        if self.collide_point(touch.x, touch.y):
            touch.ud['line'].points += [touch.x, touch.y]

    def on_touch_up(self, touch):
        if 'line' in touch.ud:
            self._line.points += touch.ud['line'].points

    def pop_line(self):
        line = self._line
        self._line = Line()
        return line
