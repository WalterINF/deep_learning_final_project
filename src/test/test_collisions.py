import unittest
from collisions import BoundingBox, Raycast
import math

class TestCollisions(unittest.TestCase):

    def test_bounding_box_collision(self):
        #testar caixas iguais
        box1 = BoundingBox(0, 0, 1, 1, 0)
        box2 = BoundingBox(0, 0, 1, 1, 0)

        self.assertTrue(box1.check_collision(box2))

        #testar caixas perfeitamente alinhadas
        box1 = BoundingBox(0, 0, 1, 1, 0)
        box2 = BoundingBox(0, 1, 1, 1, 0)

        self.assertTrue(box1.check_collision(box2))

        #testar caixas separadas
        box1 = BoundingBox(0, 0, 1, 1, 0)
        box2 = BoundingBox(0, 2, 1, 1, 0)

        self.assertFalse(box1.check_collision(box2))

        #testar caixas perfeitamente alinhadas com angulo diferente
        box1 = BoundingBox(0, 0, 1, 1, 0)
        box2 = BoundingBox(0, 1, 1, 1, math.pi/2)

        self.assertTrue(box1.check_collision(box2))

        #testar caixas separadas com angulo diferente
        box1 = BoundingBox(0, 0, 1, 1, 3*math.pi/2)
        box2 = BoundingBox(0, 2, 1, 1, math.pi/2)

        self.assertFalse(box1.check_collision(box2))

    def test_raycast_collision(self):
        raycast1 = Raycast(0, 0, 0, 10)
        box1 = BoundingBox(10, 0, 1, 1, 0)

        self.assertTrue(raycast1.check_collision(box1))

        raycast1 = Raycast(0, 0, math.pi/2, 10)
        box1 = BoundingBox(10, 0, 1, 1, math.pi/2)

        self.assertFalse(raycast1.check_collision(box1))


        



