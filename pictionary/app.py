from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button

from .agent import Agent
from .paint import PaintWidget


class PaintApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agent = None
        self.label = None
        self.label_panel = None
        self.painter = None
        self.panel = None

    def build(self):
        parent = BoxLayout(orientation='vertical')

        self.label = Label(text='None', font_size='25sp', bold=True, color=[0., 1., 1., 1.], size_hint=(1.0, 0.2))

        def set_text(word: str):
            self.label.text = word
        self.agent = Agent(set_text)

        self.painter = PaintWidget(size_hint=(0.9, 1.0))
        self.label_panel = BoxLayout(orientation='vertical', spacing=10, size_hint=(0.1, 1.))
        for i in range(10):
            def _store_image(obj, _i=i):
                self.agent.store_image(self.painter.pop_line(), str(_i))
                self.clear_canvas(obj)
            button = Button(text=str(i))
            button.bind(on_release=_store_image)
            self.label_panel.add_widget(button)

        clear_button = Button(text='Clear')
        clear_button.bind(on_release=self.clear_canvas)
        self.label_panel.add_widget(clear_button)

        accuracy_button = Button(text='Report Accuracy')
        accuracy_button.bind(on_release=self.agent.accuracy)
        self.label_panel.add_widget(accuracy_button)

        top_layout = BoxLayout(orientation='horizontal', spacing=1, size_hint=(1.0, 0.8))
        top_layout.add_widget(self.painter)
        top_layout.add_widget(self.label_panel)

        parent.add_widget(top_layout)
        parent.add_widget(self.label)

        return parent

    def clear_canvas(self, obj):
        self.painter.canvas.clear()
