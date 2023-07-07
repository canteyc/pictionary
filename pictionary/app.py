from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button

from .agent import Agent
from .paint import PaintWidget


class PaintApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.panel = None
        self.painter = None
        self.label = None
        self.agent = None

    def build(self):
        parent = BoxLayout(orientation='vertical')

        self.label = Label(text='None', color=[0., 1., 1., 1.])

        def set_text(word: str):
            self.label.text = word
        self.agent = Agent(set_text)

        self.painter = PaintWidget(self.agent.store_image, size_hint=(1., 0.8))

        self.panel = BoxLayout(orientation='horizontal', spacing=10, size_hint=(1., 0.2))
        clear_button = Button(text='Clear')
        clear_button.bind(on_release=self.clear_canvas)

        parent.add_widget(self.painter)
        parent.add_widget(self.panel)
        self.panel.add_widget(clear_button)
        self.panel.add_widget(self.label)

        return parent

    def clear_canvas(self, obj):
        self.painter.canvas.clear()
        # self.label.text = str(len(self.lines))
