from kivy.properties import NumericProperty
from kivy.uix.widget import Widget
from kivy.vector import Vector

from .ball import PongBall


class PongPaddle(Widget):
    score = NumericProperty(0)

    def bounce_ball(self, ball: PongBall):
        if self.collide_widget(ball):
            offset = (ball.center_y - self.center_y) / (self.height / 2)
            bounced: Vector = Vector(-ball.velocity_x, ball.velocity_y) * 1.25
            if abs(bounced.x) > ball.max_speed:
                bounced *= ball.max_speed / abs(bounced.x)
            ball.velocity = bounced.x, bounced.y + offset
