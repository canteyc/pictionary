from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.widget import Widget

from .paint import PaintWidget


class PaintApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.painter = None

    def build(self):
        parent = Widget()
        self.painter = PaintWidget()
        clear_button = Button(text='Clear')
        clear_button.bind(on_release=self.clear_canvas)
        parent.add_widget(self.painter)
        parent.add_widget(clear_button)
        return parent

    def clear_canvas(self, obj):
        self.painter.canvas.clear()
