from kivy.app import App
from kivy.clock import Clock

from pictionary.pong.ball import PongBall
from pictionary.pong.game import PongGame


class PongApp(App):
    def build(self):
        game_ = PongGame()
        game_.serve_ball()
        Clock.schedule_interval(game_.update, 1.0 / 60.0)
        return game_


if __name__ == '__main__':
    PongApp().run()
