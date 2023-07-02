from random import random
from typing import Callable

from kivy.graphics import Color, Line, Ellipse
from kivy.uix.widget import Widget


class PaintWidget(Widget):
    def __init__(self, callback: Callable[[Line], None], **kwargs):
        super().__init__(**kwargs)
        self.send_line = callback

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
            self.send_line(touch.ud['line'])
            image = self.export_as_image()
