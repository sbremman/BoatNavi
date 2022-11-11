import math
import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np

import gym
from gym import error, spaces
from gym.error import DependencyNotInstalled
from gym.utils import EzPickle, colorize
#from gym.utils.step_api_compatibility import step_api_compatibility

try:
    import Box2D
    from Box2D.b2 import (
        circleShape,
        contactListener,
        edgeShape,
        fixtureDef,
        polygonShape,
        revoluteJointDef,
        weldJointDef,
    )

except ImportError:
    raise DependencyNotInstalled("box2d is not installed, run `pip install gym[box2d]`")

if TYPE_CHECKING:
    import pygame


FPS = 60
SCALE = 25.0  # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6

INITIAL_RANDOM = 1000.0  # Set 1500 to make game harder


"""MAP = [['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
       ['o', 'o', 'o', 'o', 'w', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
       ['e', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
       ['o', 'o', 'o', 'o', 'w', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
       ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'w', 'o', 'o', 'o', 'o', 'o'],
       ['o', 'o', 'o', 'o', 'w', 'o', 'o', 'o', 'o', 'w', 'o', 'o', 'o', 'o', 'o'],
       ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'w', 'o', 'o', 'o', 'o', 'o'],
       ['o', 'o', 'o', 'o', 'w', 'o', 'o', 'o', 'o', 'w', 'o', 'o', 'o', 'o', 'o'],
       ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'w', 'o', 'o', 'o', 'o', 's'],
       ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'w', 'o', 'o', 'o', 'o', 'o']]"""


"""# Case 1
MAP = [['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
       ['o', 'o', 'o', 'o', 'o', 'o', 'o', 's', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
       ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
       ['w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
       ['w', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
       ['w', 'o', 'e', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
       ['w', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
       ['w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
       ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
       ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']]"""


# Case 4
MAP = [['o', 'o', 'o', 'o', 'o', 'o', 'w', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
       ['o', 'o', 'o', 'o', 'o', 'o', 'w', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
       ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
       ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
       ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
       ['e', 'o', 'o', 'o', 'o', 'o', 'w', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 's'],
       ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
       ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
       ['o', 'o', 'o', 'o', 'o', 'o', 'w', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
       ['o', 'o', 'o', 'o', 'o', 'o', 'w', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']]

VIEWPORT_W = 1200
VIEWPORT_H = 800

BLOCK_SIZE = 1.0*VIEWPORT_H / len(MAP)


#BOAT_POLY = [(-0, +12), (-10, 0), (-10, -20), (+10, -20), (+10, 0), (+0, +12)]

#BOAT_POLY = [(-20, -20), (20, -20), (20, 0), (0, 20), (-20, 0)]

"""BOAT_POLY = [(-12*BLOCK_SIZE/40, -20*BLOCK_SIZE/40), (12*BLOCK_SIZE/40, -20*BLOCK_SIZE/40),
             (13.5*BLOCK_SIZE/40, -10*BLOCK_SIZE/40), (12*BLOCK_SIZE/40, 0*BLOCK_SIZE/40),
             (7*BLOCK_SIZE/40, 10*BLOCK_SIZE/40), (2*BLOCK_SIZE/40, 15*BLOCK_SIZE/40),
             (0*BLOCK_SIZE/40, 16*BLOCK_SIZE/40), (-2*BLOCK_SIZE/40, 15*BLOCK_SIZE/40),
             (-7*BLOCK_SIZE/40, 10*BLOCK_SIZE/40), (-12*BLOCK_SIZE/40, 0*BLOCK_SIZE/40),
             (-13.5*BLOCK_SIZE/40, -10*BLOCK_SIZE/40)]"""


"""BOAT_POLY = [(-8*BLOCK_SIZE/40, -20*BLOCK_SIZE/40), (8*BLOCK_SIZE/40, -20*BLOCK_SIZE/40),
             (8.5*BLOCK_SIZE/40, -10*BLOCK_SIZE/40), (8*BLOCK_SIZE/40, 0*BLOCK_SIZE/40),
             (3*BLOCK_SIZE/40, 10*BLOCK_SIZE/40), (1*BLOCK_SIZE/40, 12*BLOCK_SIZE/40),
             (0*BLOCK_SIZE/40, 13*BLOCK_SIZE/40), (-1*BLOCK_SIZE/40, 12*BLOCK_SIZE/40),
             (-3*BLOCK_SIZE/40, 10*BLOCK_SIZE/40), (-8*BLOCK_SIZE/40, 0*BLOCK_SIZE/40),
             (-8.5*BLOCK_SIZE/40, -10*BLOCK_SIZE/40)]"""


BOAT_POLY = [(-6*BLOCK_SIZE/40, -20*BLOCK_SIZE/40), (6*BLOCK_SIZE/40, -20*BLOCK_SIZE/40),
             (7.2*BLOCK_SIZE/40, -5*BLOCK_SIZE/40), (7.5*BLOCK_SIZE/40, 0*BLOCK_SIZE/40),
             (7.7*BLOCK_SIZE/40, 5*BLOCK_SIZE/40), (6*BLOCK_SIZE/40, 10*BLOCK_SIZE/40),
             (4*BLOCK_SIZE/40, 15*BLOCK_SIZE/40), (0*BLOCK_SIZE/40, 19*BLOCK_SIZE/40),
             (-4*BLOCK_SIZE/40, 15*BLOCK_SIZE/40), (-6*BLOCK_SIZE/40, 10*BLOCK_SIZE/40),
             (-7.7*BLOCK_SIZE/40, 5*BLOCK_SIZE/40), (-7.5*BLOCK_SIZE/40, 0*BLOCK_SIZE/40),
             (-7.2*BLOCK_SIZE/40, -5*BLOCK_SIZE/40)]

BOAT_HOUSE_POLY = [(-4*BLOCK_SIZE/40, -7.5*BLOCK_SIZE/40), (4*BLOCK_SIZE/40, -7.5*BLOCK_SIZE/40),
             (5*BLOCK_SIZE/40, -6.5*BLOCK_SIZE/40), (5*BLOCK_SIZE/40, 6.5*BLOCK_SIZE/40),
             (4*BLOCK_SIZE/40, 7.5*BLOCK_SIZE/40), (-4*BLOCK_SIZE/40, 7.5*BLOCK_SIZE/40),
             (-5*BLOCK_SIZE/40, 6.5*BLOCK_SIZE/40), (-5*BLOCK_SIZE/40, -6.5*BLOCK_SIZE/40)]



#BOX_POLY = [(-20, +20), (20, 20), (20, -20), (-20, -20)]

#COLOR_BLUE = (0, 114, 178)
#COLOR_VERMILLION = (213, 94, 0)
#COLOR_BLUEGREEN = (0, 158, 115)
#COLOR_REDDISHPURPLE = (204, 121, 167)
#COLOR_SKYBLUE = (86, 180, 233)
#COLOR_ORANGE = (230, 159, 0)
#COLOR_YELLOW = (240, 228, 66)

COLOR_BLACK = (26, 28, 44)
COLOR_PURPLE = (93, 39, 93)
COLOR_RED = (177, 62, 83)
COLOR_ORANGE = (239, 125, 87)
COLOR_YELLOW = (255, 205, 117)
COLOR_LGREEN = (167, 240, 112)
COLOR_GREEN = (56, 183, 100)
COLOR_DGREEN = (37, 113, 121)
COLOR_DBLUE = (41, 54, 111)
COLOR_BLUE = (59, 93, 201)
COLOR_LBLUE = (65, 166, 246)
COLOR_CYAN = (115, 239, 247)
COLOR_WHITE = (244, 244, 244)
COLOR_LGREY = (148, 176, 194)
COLOR_GREY = (86, 108, 134)
COLOR_DGREY = (51, 60, 87)

class BoatPart:
    pass

boat_middle = BoatPart()
boat_middle.shape = polygonShape(vertices=[(x*1.2 / (SCALE*1.1), y*1.2 / (SCALE*1.1)) for x, y in BOAT_POLY])
boat_middle.color1 = COLOR_LGREY
boat_middle.color2 = COLOR_BLACK
boat_middle.pos_rel_to_body = (0, -2.0/SCALE)

boat_inner = BoatPart()
boat_inner.shape = polygonShape(vertices=[(x*1.2 / (SCALE*1.2), y*1.2 / (SCALE*1.2)) for x, y in BOAT_POLY])
boat_inner.color1 = COLOR_GREY
boat_inner.color2 = COLOR_BLACK
boat_inner.pos_rel_to_body = (0, -3.0/SCALE)

boat_window = BoatPart()
boat_window.shape = polygonShape(vertices=[(x*1.2 / (SCALE * 1.1), y*1.2 / (SCALE * 5)) for x, y in BOAT_HOUSE_POLY])
boat_window.color1 = COLOR_DBLUE
boat_window.color2 = COLOR_BLACK
boat_window.pos_rel_to_body = (0, 11/SCALE)













class BoatNavi(gym.Env):
    """
        ### Description

        ### Action Space



        ### Observation Space

        ### Rewards

        ### Starting State

        ### Episode Termination

        ### Arguments

        ### Version History
        - v2: Count energy spent and in v0.24, added turbulance with wind power and turbulence_power parameters
        - v1: Legs contact with ground added in state vector; contact with ground
            give +10 reward points, and -10 if then lose contact; reward
            renormalized to 200; harder initial random push.
        - v0: Initial version
        <!-- ### References -->
        ### Credits
        Created by Sindre B. Stenrud Remman
        """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(
            self,
            render_mode: Optional[str] = None,
            continuous: bool = False,
            gravity: float = -10.0,
            enable_wind: bool = False,
            wind_power: float = 15.0,
            turbulence_power: float = 1.5,
    ):

        self.screen: pygame.Surface = None
        self.clock = None
        self.isopen = True
        self.world = Box2D.b2World(gravity=(0,0))
        self.boat: Optional[Box2D.b2Body] = None

        self.prev_reward = None

        self.render_mode = render_mode
        self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float64)
        self.goal = None
        self.start = None

        self.boat_position = None

        self.boat_draw_list = [boat_middle, boat_inner, boat_window]

    def _destroy(self):
        self.world.DestroyBody(self.boat)
        self.boat = None

    def reset(self, initial_plan=None):
        #super().reset(seed=seed)
        #self._destroy()
        #self.world.contactListener_keepref = ContactDetector(self)
        #self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        self.plan = initial_plan

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        # ----------- Build map ----------- #

        map_unit_w = VIEWPORT_W / SCALE / len(MAP[0])
        map_unit_h = VIEWPORT_H / SCALE / len(MAP)
        map_list = []

        # Obstacles

        scaling = BLOCK_SIZE/40

        for i in range(len(MAP)):
            for j in range(len(MAP[i])):
                if MAP[i][j] == 'w':
                    end = self.world.CreateStaticBody(
                        position=((0.5+j)*map_unit_w, (9.5-i)*map_unit_h),
                        shapes=polygonShape(box=(20*scaling/SCALE, 20*scaling/SCALE))
                    )
                    end.color1 = COLOR_ORANGE
                    end.color2 = COLOR_BLACK

                    map_list.append(end)

                    """if i+1 >= len(MAP):
                        end = self.world.CreateStaticBody(
                            position=((0.5+j)*map_unit_w, (9.5-i)*map_unit_h-13/SCALE),
                            shapes=polygonShape(box=(20/SCALE, 7/SCALE))
                        )
                        end.color1 = COLOR_RED
                        end.color2 = COLOR_BLACK

                        map_list.append(end)"""

                    if i+1 >= len(MAP) or MAP[i+1][j] != 'w':

                        end = self.world.CreateStaticBody(
                            position=((0.5+j)*map_unit_w, (9.5-i)*map_unit_h-13*scaling/SCALE),
                            shapes=polygonShape(box=(20*scaling/SCALE, 7*scaling/SCALE))
                        )
                        end.color1 = COLOR_RED
                        end.color2 = COLOR_BLACK

                        map_list.append(end)


                    """end = self.world.CreateStaticBody(
                        position=((0.5+j)*map_unit_w, (9.5-i)*map_unit_h),
                        shapes=polygonShape(box=(2/SCALE, 2/SCALE))
                    )
                    end.color1 = COLOR_YELLOW
                    end.color2 = COLOR_YELLOW

                    map_list.append(end)"""

                elif MAP[i][j] == 'e':
                    self.end = ((0.5+j)*map_unit_w, (9.5-i)*map_unit_h)

                    end = self.world.CreateStaticBody(
                        position=self.end,
                        shapes=polygonShape(box=(20*scaling/SCALE, 20*scaling/SCALE))
                    )
                    end.color1 = COLOR_CYAN
                    end.color2 = COLOR_BLACK
                    end.fixtures[0].sensor = True

                    map_list.append(end)

                    end = self.world.CreateStaticBody(
                        position=((0.5+j)*map_unit_w, (9.5-i)*map_unit_h-13*scaling/SCALE),
                        shapes=polygonShape(box=(20*scaling/SCALE, 8*scaling/SCALE))
                    )
                    end.color1 = COLOR_LBLUE
                    end.color2 = COLOR_BLACK
                    end.fixtures[0].sensor = True

                    map_list.append(end)

                elif MAP[i][j] == 's':

                    self.start = ((0.5+j)*map_unit_w, (9.5-i)*map_unit_h)
                    break

                    end = self.world.CreateStaticBody(
                        position=self.start,
                        shapes=polygonShape(box=(20*scaling/SCALE, 20*scaling/SCALE))
                    )
                    end.color1 = COLOR_LGREEN
                    end.color2 = COLOR_BLACK
                    end.fixtures[0].sensor = True

                    map_list.append(end)

                    end = self.world.CreateStaticBody(
                        position=((0.5+j)*map_unit_w, (9.5-i)*map_unit_h-13*scaling/SCALE),
                        shapes=polygonShape(box=(20*scaling/SCALE, 8*scaling/SCALE))
                    )
                    end.color1 = COLOR_GREEN
                    end.color2 = COLOR_BLACK
                    end.fixtures[0].sensor = True

                    map_list.append(end)

                else:
                    pass

        # Wall around

        """grey_color = COLOR_BLACK

        for i in range(2):
            wall = self.world.CreateStaticBody(
                position=(W / 2, H*i),
                shapes=polygonShape(box=(W, 1 / SCALE))
            )
            wall.color1 = grey_color
            wall.color2 = grey_color

            map_list.append(wall)

        for i in range(2):
            wall = self.world.CreateStaticBody(
                position=(W * i, H / 2),
                shapes=polygonShape(box=(1 / SCALE, H))
            )
            wall.color1 = grey_color
            wall.color2 = grey_color

            map_list.append(wall)"""

        initial_x = self.start[0]
        initial_y = self.start[1]
        initial_angle = 1.0*math.pi/2

        boat_fixture_outer = fixtureDef(
                shape=polygonShape(
                    vertices=[(x*1.2 / SCALE, y*1.2 / SCALE) for x, y in BOAT_POLY]
                ),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.1,
            )

        boat_fixture_motor = fixtureDef(
            shape=polygonShape(box=(2*scaling/SCALE, 4*scaling/SCALE)),
            density=0.1,
            friction=0.1,
            categoryBits=0x0010,
            maskBits=0x001,)

        self.boat: Box2D.b2Body = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=initial_angle,
            fixtures=boat_fixture_outer,
            angularDamping=1.5,
            linearDamping=0.7,
        )

        self.boat.color1 = COLOR_WHITE
        self.boat.color2 = COLOR_BLACK
        self.boat_list = [self.boat]
        self.boat_position = [self.boat.position.x, self.boat.position.y]

        """boat_motor = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=(0.0),
            fixtures=boat_fixture_motor,
        )

        boat_motor.color1 = COLOR_GREY
        boat_motor.color2 = COLOR_BLACK

        rjd = revoluteJointDef(
            bodyA=self.boat,
            bodyB=boat_motor,
            localAnchorA=(0, -20*scaling / SCALE),
            localAnchorB=(0, 3*scaling / SCALE),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=10,
            motorSpeed=0.3,  # low enough not to jump back into the sky
        )

        rjd.lowerAngle = -0.5  # The most esoteric numbers here, angled legs have freedom to travel within
        rjd.upperAngle = 0.5

        boat_motor.joint = self.world.CreateJoint(rjd)
        boat_list.append(boat_motor)"""



        self.drawlist = map_list+self.boat_list

        if self.render_mode == "human":
            self.render()



        return self.step(np.array([0.0]*self.action_space.shape[0]), plan=initial_plan)

    def step(self, action, optional=0.0, plan=None):
        assert self.boat is not None, "You forgot to call reset()"

        self.plan = plan

        state = [None]
        reward = 0
        terminated = None

        dir_test = dir(self.boat)

        forward_force = (-action[1]*math.sin(self.boat.angle), action[1]*math.cos(self.boat.angle))

        self.boat.ApplyTorque(action[0]*20, True)

        self.boat.ApplyLinearImpulse(
            forward_force,
            (self.boat.position.x, self.boat.position.y),
            True,
        )
        # self.boat.transform(self.boat.position+[0.1,0.0])
        test = self.boat.transform
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        """self.boat.position.x, self.boat.position.y = action[0], action[1]
        self.boat.angle = optional #-math.atan2(action[0] - self.boat_position[0], action[1] - self.boat_position[1])

        self.boat_position = [self.boat.position.x, self.boat.position.y]"""

        return np.array(state, dtype=np.float32), reward, terminated, False, {}

    def render(self, mode="human"):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[box2d]`"
            )

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))

        pygame.transform.scale(self.surf, (SCALE, SCALE))
        pygame.draw.rect(self.surf, COLOR_BLUE, self.surf.get_rect())

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color1,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color2,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )

                else:
                    path = [trans * v * SCALE for v in f.shape.vertices]
                    pygame.draw.polygon(self.surf, color=obj.color1, points=path)
                    gfxdraw.aapolygon(self.surf, path, obj.color1)
                    pygame.draw.aalines(
                        self.surf, color=obj.color2, points=path, closed=True
                    )


        for obj in self.boat_draw_list:
            obj_trans = self.boat.transform
            angle = obj_trans.angle
            #dir_test = dir(obj_trans)
            obj_trans.position.x += math.cos(angle)*obj.pos_rel_to_body[0] - math.sin(angle)*obj.pos_rel_to_body[1]
            obj_trans.position.y += math.sin(angle) * obj.pos_rel_to_body[0] + math.cos(angle) * obj.pos_rel_to_body[1]
            path = [obj_trans * v * SCALE for v in obj.shape.vertices]
            pygame.draw.polygon(self.surf, color=obj.color1, points=path)
            gfxdraw.aapolygon(self.surf, path, obj.color1)
            pygame.draw.aalines(
                self.surf, color=obj.color2, points=path, closed=True
            )


        #Draw plan arrow
        width = 5
        color = COLOR_BLACK
        if self.plan is not None:
            last_point = self.plan[-1]
            next_last_point = self.plan[-10]
            pygame.draw.lines(self.surf, color=COLOR_BLACK, points=self.plan[:-4]*SCALE, closed=False, width=width)

            angle_at_end = math.atan2(last_point[0]-next_last_point[0], last_point[1]-next_last_point[1])

            perpendicular_vector = np.array([math.cos(angle_at_end), -math.sin(angle_at_end)])
            paralell_vector = np.array([-math.sin(angle_at_end), -math.cos(angle_at_end)])

            arrow_head_coord = (SCALE*last_point, SCALE*(last_point+0.2 * width * paralell_vector + 0.1 * width * perpendicular_vector),
                                SCALE*(last_point+0.2 * width* paralell_vector - 0.1 * width * perpendicular_vector))


            pygame.draw.polygon(self.surf, COLOR_BLACK, arrow_head_coord)
            """pygame.draw.aalines(
                self.surf, color=color, points=arrow_head_coord, closed=True
            )"""





        self.surf = pygame.transform.flip(self.surf, False, True)

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )

    def spawn_obstacle_boat(self, coordinates, initial_impulse, initial_angle):

        boat_fixture_outer = fixtureDef(
            shape=polygonShape(
                vertices=[(x * 1.2 / SCALE, y * 1.2 / SCALE) for x, y in BOAT_POLY]
            ),
            density=5.0,
            friction=0.1,
            categoryBits=0x0010,
            maskBits=0x001,  # collide only with ground
            restitution=0.1,
        )

        obstacle_boat: Box2D.b2Body = self.world.CreateDynamicBody(
            position=(coordinates[0], coordinates[1]),
            angle=initial_angle,
            fixtures=boat_fixture_outer,
            angularDamping=100000.0,
            linearDamping=0.0,
        )

        obstacle_boat.color1 = COLOR_BLACK
        obstacle_boat.color2 = COLOR_WHITE
        obstacle_boat_list = obstacle_boat

        forward_force = (-initial_impulse * math.sin(initial_angle), initial_impulse * math.cos(initial_angle))
        obstacle_boat.ApplyLinearImpulse(
            forward_force,
            (self.boat.position.x, self.boat.position.y),
            True,
        )

        self.drawlist.append(obstacle_boat_list)


class LiftingAtan2:

    def __init__(self, initial_angle):

        self.last_angle = initial_angle
        self.num_rotations = 0

    def __call__(self, y, x):

        angle = math.atan2(y, x)
        if (math.copysign(1, angle) != math.copysign(1, self.last_angle)) and abs(angle) > math.pi/2:
            if math.copysign(1, angle) > 0.0:
                self.num_rotations -= 1

            else:
                self.num_rotations += 1

        self.last_angle = angle

        angle = 2*math.pi*self.num_rotations+angle

        return angle


if __name__ == '__main__':
    from scipy import interpolate

    #demo_ship(BoatNavi(render_mode="human"), render=True)
    playing = False

    env = BoatNavi(render_mode="human")

    observation = env.reset()

    #actions = np.array([[10.0,2.5]]*800)

    #size = 1000
    #y = [[14.4, 8], [24, 26], [27.2, 24]]
    """while True:
        import keyboard
        if keyboard.is_pressed('esc'):
            break"""


    end_y = env.end[1]
    start_y = env.start[1]

    end_x = env.end[0]
    start_x = env.start[0]


    #1 - follows plan

    #x = np.array([start_x, 20, 30, end_x])
    #y = np.array([start_y, 16, 18, end_y])

    #x = np.array([start_x, 12, 30, end_x])
    #y = np.array([start_y, 17, 20, end_y])

    # Define some points:
    """points = np.array([[start_x, 35, 24, end_x],
                       [start_y, 20, 15, end_y]]).T

    points = np.array([[start_x, 30, 20, end_x],
                       [start_y, 18, 16, end_y]]).T""" # ,

    points = np.array([[start_x, 30, 26, 7.44249, end_x],
                       [start_y, 12, 19.5, 20.0953, end_y]]).T  # a (nbre_points x nbre_dim) array

    # Linear length along the line:
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0) / distance[-1]

    alpha = np.linspace(0, 1, 1500)

    """interpolator = interpolate.interp1d(distance, points, kind='quadratic', axis=0)
    positions = interpolator(alpha)"""

    # Build a list of the spline function, one for each dimension:
    splines = [interpolate.UnivariateSpline(distance, coords, k=2, s=.9) for coords in points.T]
    # Computed the spline for the asked distances:
    positions = np.vstack(spl(alpha) for spl in splines).T



    # "fake path"

    points = np.array([[start_x, 30, 26, end_x],
                       [start_y, 11, 10, end_y]]).T  # a (nbre_points x nbre_dim) array

    # Linear length along the line:
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0) / distance[-1]

    alpha = np.linspace(0, 1, 1500)

    """interpolator = interpolate.interp1d(distance, points, kind='quadratic', axis=0)
    positions = interpolator(alpha)"""

    # Build a list of the spline function, one for each dimension:
    splines = [interpolate.UnivariateSpline(distance, coords, k=2, s=.9) for coords in points.T]
    # Computed the spline for the asked distances:
    fake_positions = np.vstack(spl(alpha) for spl in splines).T






    #f = interpolate.interp1d(y, x, kind='cubic')
    #f = interpolate.UnivariateSpline(y, x)
    #xnew = np.linspace(end_y, start_y, 500)

    #positions_old = np.array([[f(xnew[i]).item(), xnew[i]] for i in range(xnew.shape[0])])
    #positions = np.array([[pos[0], pos[1]] for pos in interpolated_points])

    angles = [-math.atan2(positions[1][0] - positions[0][0], positions[1][1] - positions[0][1])]
    velocities = []


    atan2 = LiftingAtan2(-angles[0])

    for i in range(1, len(positions)):
        angle = -atan2(positions[i][0] - positions[i-1][0], positions[i][1] - positions[i-1][1])

        angles.append(angle)

    for i in range(len(positions)-1):
        velocity = math.sqrt((positions[i+1][0] - positions[i][0])**2 + (positions[i+1][1] - positions[i][1])**2)

        velocities.append(velocity)

    for i in range(420):
        angles.append(angles[-1])

        velocities.append(velocities[-1])



    #positions = [fun_follow_plan(x) for x in np.arange(27.2, 14.4, -0.05)]





    if playing:
        import keyboard

        while True:
            action = np.array([0.0, 0.0])

            if keyboard.is_pressed('w'):
                action[1] += 0.5

            if keyboard.is_pressed('s'):
                action[1] -= 0.5

            if keyboard.is_pressed('a'):
                action[0] += 0.5

            if keyboard.is_pressed('d'):
                action[0] -= 0.5

            if keyboard.is_pressed('esc'):
                break

            observation, reward, terminated, truncated, info = env.step(action*2)
            env.render()

            print(env.boat.position)
            #print(action)



    else:
        desired_speed = 2
        Kp_speed = 10
        Kp_angle = 10
        Kp_xte = 1
        desired_angle_hist = []
        curr_angle_hist = []
        import copy

        num_points = 10

        real_positions = []

        test = 1


        obstacle_coord = [-22, 10]
        obstacle_impulse = 90
        obstacle_angle = -math.pi/2
        env.spawn_obstacle_boat(obstacle_coord, obstacle_impulse, obstacle_angle)

        for j in range(len(angles)):
            real_positions.append(np.array([env.boat.position.x, env.boat.position.y]))

            #end_y = 14.4
            start_y = env.boat.position.y

            #end_x = 8.0
            start_x = env.boat.position.x

            length_ahead = 0.5

            ahead_x = start_x - math.sin(env.boat.angle) * length_ahead
            ahead_y = start_y + math.cos(env.boat.angle) * length_ahead



            #x = np.array([start_x, positions[j+diff_points][0], positions[j+2*diff_points][0], positions[j+200][0], end_x])
            #y = np.array([start_y, positions[j+diff_points][1], positions[j+2*diff_points][1], positions[j+200][1], end_y])

            start_change = 330
            stop_change = start_change+10

            try:
                if j < start_change:
                    diff_points = (len(fake_positions) - j) // num_points
                    x = [fake_positions[j + k * diff_points][0] for k in range(num_points)]
                    y = [fake_positions[j + k * diff_points][1] for k in range(num_points)]
                    x[0] = start_x
                    y[0] = start_y

                    #x.insert(1, ahead_x)
                    #y.insert(1, ahead_y)
                    # x[1] = ahead_x
                    # y[1] = ahead_y
                    x.append(end_x)
                    y.append(end_y)

                    x = np.array(x)
                    y = np.array(y)
                    k = 2
                    print(j)

                elif j < stop_change:
                    diff_points = (len(fake_positions) - j) // num_points
                    x = [fake_positions[j + k * diff_points][0] for k in range(num_points)]
                    y = [fake_positions[j + k * diff_points][1] for k in range(num_points)]
                    x[0] = start_x
                    y[0] = start_y

                    x.insert(1, ahead_x)
                    y.insert(1, ahead_y)
                    # x[1] = ahead_x
                    # y[1] = ahead_y
                    x.append(end_x)
                    y.append(end_y)

                    x_fake = np.array(x)
                    y_fake = np.array(y)

                    diff_points = (len(positions) - j) // num_points
                    x = [positions[j + k * diff_points][0] for k in range(num_points)]
                    y = [positions[j + k * diff_points][1] for k in range(num_points)]
                    x[0] = start_x
                    y[0] = start_y

                    x.insert(1, ahead_x)
                    y.insert(1, ahead_y)
                    #x[1] = ahead_x
                    #y[1] = ahead_y
                    x.append(end_x)
                    y.append(end_y)

                    x_real = np.array(x)
                    y_real = np.array(y)
                    k=2
                    a = np.array([[1, start_change], [1, stop_change]])
                    b = np.array([0.0, 1.0])
                    x0, x1 = np.linalg.solve(a, b)

                    proportion = x0 + x1*j

                    x = x_fake * (1-proportion) + x_real * (proportion)
                    y = y_fake * (1 - proportion) + y_real * (proportion)



                else:
                    diff_points = (len(positions) - j) // num_points
                    x = [positions[j + k * diff_points][0] for k in range(num_points)]
                    y = [positions[j + k * diff_points][1] for k in range(num_points)]
                    x[0] = start_x
                    y[0] = start_y

                    x.insert(1, ahead_x)
                    y.insert(1, ahead_y)
                    # x[1] = ahead_x
                    # y[1] = ahead_y
                    x.append(end_x)
                    y.append(end_y)

                    x = np.array(x)
                    y = np.array(y)
                    k = 2



            except:
                diff_points = (len(new_positions)) // num_points
                x = [new_positions[k * diff_points][0] for k in range(num_points)]
                y = [new_positions[k * diff_points][1] for k in range(num_points)]
                x[0] = start_x
                y[0] = start_y

                x.insert(1, ahead_x)
                y.insert(1, ahead_y)
                x.append(end_x)
                y.append(end_y)

                x = np.array(x)
                y = np.array(y)
                k = 2
                # f = interpolate.interp1d(y, x, kind='cubic')

            #xnew = np.linspace(end_y, start_y, 500)

            #new_positions = np.array([[f(xnew[i]).item(), xnew[i]] for i in range(xnew.shape[0])])

            # Define some points:
            points = np.array([x,
                               y]).T  # a (nbre_points x nbre_dim) array

            # Linear length along the line:
            distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
            distance = np.insert(distance, 0, 0) / distance[-1]

            alpha = np.linspace(0, 1, 500)

            """interpolator = interpolate.interp1d(distance, points, kind='quadratic', axis=0)
            positions = interpolator(alpha)"""

            # Build a list of the spline function, one for each dimension:
            splines = [interpolate.UnivariateSpline(distance, coords, k=k, s=.1) for coords in points.T]
            # Computed the spline for the asked distances:
            new_positions = np.vstack(spl(alpha) for spl in splines).T


            #observation, reward, terminated, truncated, info = env.step(positions[j], angles[j])
            curr_speed = env.boat.linearVelocity.length
            #print(curr_speed)
            #desired_speed = velocities[j]*FPS
            error_speed = desired_speed-curr_speed
            curr_angle = env.boat.angle
            desired_angle = angles[j]
            desired_angle_hist.append(desired_angle)
            curr_angle_hist.append(curr_angle)
            error_angle = desired_angle-curr_angle

            #Calculate cross-track error

            diff_pos = env.boat.position-positions
            ind_closest_point = np.argmin(np.linalg.norm(diff_pos, axis=1))
            xte = np.linalg.norm(diff_pos[ind_closest_point])

            distance_vector_angle = -math.atan2(diff_pos[ind_closest_point][1], diff_pos[ind_closest_point][0])-curr_angle
            sign_xte = -math.copysign(1, distance_vector_angle)


            if error_angle > 2:
                curr_angle_hist = curr_angle_hist[-10:]
                desired_angle_hist = desired_angle_hist[-10:]
                print(error_angle)

            """plot_positions = copy.deepcopy(positions[j:])
            plot_positions[0] = env.boat.position"""

            env.step([Kp_angle * error_angle + Kp_xte*sign_xte*xte, Kp_speed * error_speed], plan=new_positions)



            env.render()

        real_positions = np.array(real_positions)
        env.reset()
        for j in range(len(angles)):
            """
            start_y = 14.4
            end_y = env.boat.position.y

            start_x = 8.0
            end_x = env.boat.position.x

            diff_points = (len(positions)-j)//num_points

            #x = np.array([start_x, positions[j+diff_points][0], positions[j+2*diff_points][0], positions[j+200][0], end_x])
            #y = np.array([start_y, positions[j+diff_points][1], positions[j+2*diff_points][1], positions[j+200][1], end_y])

            try:
                x = [positions[j + k * diff_points][0] for k in range(num_points)]
                y = [positions[j + k * diff_points][1] for k in range(num_points)]
                x[0] = end_x
                y[0] = end_y
                x.append(start_x)
                y.append(start_y)

                x = np.array(x)
                y = np.array(y)
                f = interpolate.interp1d(y, x, kind='cubic')

            except:
                x = np.array([end_x, start_x])
                y = np.array([end_y, start_y])
                f = interpolate.interp1d(y, x)

            xnew = np.linspace(end_y, start_y, 500)

            new_positions = np.array([[f(xnew[i]).item(), xnew[i]] for i in range(xnew.shape[0])])"""


            #observation, reward, terminated, truncated, info = env.step(positions[j], angles[j])
            curr_speed = env.boat.linearVelocity.length
            #print(curr_speed)
            #desired_speed = velocities[j]*FPS
            error_speed = desired_speed-curr_speed
            curr_angle = env.boat.angle
            desired_angle = angles[j]
            desired_angle_hist.append(desired_angle)
            curr_angle_hist.append(curr_angle)
            error_angle = desired_angle-curr_angle

            #Calculate cross-track error

            diff_pos = env.boat.position-positions
            ind_closest_point = np.argmin(np.linalg.norm(diff_pos, axis=1))
            xte = np.linalg.norm(diff_pos[ind_closest_point])

            distance_vector_angle = -math.atan2(diff_pos[ind_closest_point][1], diff_pos[ind_closest_point][0])-curr_angle
            sign_xte = -math.copysign(1, distance_vector_angle)


            if error_angle > 2:
                curr_angle_hist = curr_angle_hist[-10:]
                desired_angle_hist = desired_angle_hist[-10:]
                print(error_angle)

            """plot_positions = copy.deepcopy(positions[j:])
            plot_positions[0] = env.boat.position"""

            env.step([Kp_angle * error_angle + Kp_xte*sign_xte*xte, Kp_speed * error_speed], plan=real_positions[j:])



            env.render()


        """i = 0

        while True:
            action = env.action_space.sample()
            action = np.array([0.0, 0.0])
            action = actions[i]
            i += 1

            observation, reward, terminated, truncated, info = env.step(action)
            env.render()

            if terminated or truncated:
                observation, info = env.reset()"""


    env.close()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
