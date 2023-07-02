from kivy.properties import ObjectProperty
from kivy.uix.widget import Widget

from .ball import PongBall
from .paddle import PongPaddle


class PongGame(Widget):
    ball: PongBall = ObjectProperty(None)
    left_paddle: PongPaddle = ObjectProperty(None)
    right_paddle: PongPaddle = ObjectProperty(None)

    def serve_ball(self, velocity=(4, 0)):
        self.ball.center = self.center
        self.ball.velocity = velocity

    def update(self, dt):
        self.ball.move()

        # bounce off paddles
        self.left_paddle.bounce_ball(self.ball)
        self.right_paddle.bounce_ball(self.ball)

        # bounce off bottom or top
        if self.ball.y < 0 or self.ball.top > self.height:
            self.ball.velocity_y *= -1

        # score points
        if self.ball.x < 0:
            self.left_paddle.score += 1
            if self.left_paddle.score >= 4:
                self.winner("left")
            self.serve_ball(velocity=(4, 0))
        elif self.ball.right > self.width:
            self.right_paddle.score += 1
            if self.right_paddle.score >= 4:
                self.winner("right")
            self.serve_ball(velocity=(-4, 0))

    def on_touch_move(self, touch):
        if touch.x < self.width / 3:
            self.left_paddle.center_y = touch.y
        if touch.x > self.width - self.width / 3:
            self.right_paddle.center_y = touch.y

    def winner(self, side):
        print(f'{side} wins!')
        self.get_root_window().close()
