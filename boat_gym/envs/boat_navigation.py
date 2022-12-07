import math
from typing import TYPE_CHECKING, Optional

import numpy as np

import gym
from gym import spaces
from gym.error import DependencyNotInstalled

try:
    import Box2D
    from Box2D.b2 import (
        circleShape,
        fixtureDef,
        polygonShape,
        revoluteJointDef,
    )


except ImportError:
    raise DependencyNotInstalled("box2d is not installed, run `pip install gym[box2d]`")

if TYPE_CHECKING:
    import pygame


FPS = 60
SCALE = 25.0  # affects how fast-paced the game is, forces should be adjusted as well


MAP = [['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
       ['o', 'o', 'o', 'o', 'w', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
       ['e', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
       ['o', 'o', 'o', 'o', 'w', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
       ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'w', 'o', 'o', 'o', 'o', 'o'],
       ['o', 'o', 'o', 'o', 'w', 'o', 'o', 'o', 'o', 'w', 'o', 'o', 'o', 'o', 's'],
       ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'w', 'o', 'o', 'o', 'o', 'o'],
       ['o', 'o', 'o', 'o', 'w', 'o', 'o', 'o', 'o', 'w', 'o', 'o', 'o', 'o', 'o'],
       ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'w', 'o', 'o', 'o', 'o', 'o'],
       ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'w', 'o', 'o', 'o', 'o', 'o']]


VIEWPORT_W = 1200
VIEWPORT_H = 800

BLOCK_SIZE = 1.0*VIEWPORT_H / len(MAP)


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



class BoatNaviEnv(gym.Env):
    """
        ### Description

        ### Action Space



        ### Observation Space

        ### Rewards

        ### Starting State

        ### Episode Termination

        ### Arguments

        ### Version History
        - v0: Initial version
        <!-- ### References -->
        ### Credits
        Created by Sindre B. Stenrud Remman
        """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
        "observation_types": ["kinematic", "pixel", "both"]
    }

    def __init__(
            self,
            render_mode: Optional[str] = None,
            observation_type: Optional[str] = "kinematic",

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
        self.debug_points = []

        self.observation_type = observation_type

    def _destroy(self):
        self.world.DestroyBody(self.boat)
        self.boat = None

    def reset(self, initial_plan=None):
        self.game_over = False
        self.prev_shaping = None
        self.map = np.zeros([len(MAP), len(MAP[0])])

        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                # 0 = water
                # 1 = obstacle
                # 2 = goal
                # 3 = boat

                if MAP[i][j] == 'o':
                    self.map[i][j] = 0

                elif MAP[i][j] == 'w':
                    self.map[i][j] = 1

                elif MAP[i][j] == 'e':
                    self.map[i][j] = 2



        self.plan = initial_plan

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        # ----------- Build map ----------- #

        self.map_unit_w = VIEWPORT_W / SCALE / len(MAP[0])
        self.map_unit_h = VIEWPORT_H / SCALE / len(MAP)
        map_list = []

        # Obstacles

        scaling = BLOCK_SIZE/40

        for i in range(len(MAP)):
            for j in range(len(MAP[i])):
                if MAP[i][j] == 'w':
                    end = self.world.CreateStaticBody(
                        position=((0.5+j)*self.map_unit_w, (9.5-i)*self.map_unit_h),
                        shapes=polygonShape(box=(20*scaling/SCALE, 20*scaling/SCALE))
                    )
                    end.color1 = COLOR_ORANGE
                    end.color2 = COLOR_BLACK

                    map_list.append(end)

                    if i+1 >= len(MAP) or MAP[i+1][j] != 'w':

                        end = self.world.CreateStaticBody(
                            position=((0.5+j)*self.map_unit_w, (9.5-i)*self.map_unit_h-13*scaling/SCALE),
                            shapes=polygonShape(box=(20*scaling/SCALE, 7*scaling/SCALE))
                        )
                        end.color1 = COLOR_RED
                        end.color2 = COLOR_BLACK

                        map_list.append(end)

                elif MAP[i][j] == 'e':
                    self.end = ((0.5+j)*self.map_unit_w, (9.5-i)*self.map_unit_h)

                    end = self.world.CreateStaticBody(
                        position=self.end,
                        shapes=polygonShape(box=(20*scaling/SCALE, 20*scaling/SCALE))
                    )
                    end.color1 = COLOR_CYAN
                    end.color2 = COLOR_BLACK
                    end.fixtures[0].sensor = True

                    map_list.append(end)

                    end = self.world.CreateStaticBody(
                        position=((0.5+j)*self.map_unit_w, (9.5-i)*self.map_unit_h-13*scaling/SCALE),
                        shapes=polygonShape(box=(20*scaling/SCALE, 8*scaling/SCALE))
                    )
                    end.color1 = COLOR_LBLUE
                    end.color2 = COLOR_BLACK
                    end.fixtures[0].sensor = True

                    map_list.append(end)

                elif MAP[i][j] == 's':

                    self.start = ((0.5+j)*self.map_unit_w, (9.5-i)*self.map_unit_h)
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
                        position=((0.5+j)*self.map_unit_w, (9.5-i)*self.map_unit_h-13*scaling/SCALE),
                        shapes=polygonShape(box=(20*scaling/SCALE, 8*scaling/SCALE))
                    )
                    end.color1 = COLOR_GREEN
                    end.color2 = COLOR_BLACK
                    end.fixtures[0].sensor = True

                    map_list.append(end)

                else:
                    pass

        # Wall around

        grey_color = COLOR_BLACK

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

            map_list.append(wall)

        initial_x = self.start[0]
        initial_y = self.start[1]
        initial_angle = math.pi/4

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
            linearDamping=0.5,
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

    def step(self, action, optional=0.0, plan=None, debug_points=[]):
        assert self.boat is not None, "You forgot to call reset()"

        self.plan = plan
        self.debug_points = debug_points

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

        if self.observation_type == "pixel":
            # 0 = water
            # 1 = obstacle
            # 2 = goal
            # 3 = boat

            state = np.copy(self.map)
            #print()
            x = math.floor(self.boat.position.x/self.map_unit_w)
            y = 9-math.floor(self.boat.position.y/self.map_unit_h)
            state[y][x] = 3



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

        for debug_point in self.debug_points:
            pygame.draw.circle(self.surf, COLOR_RED, debug_point*SCALE, 4)






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

    playing = True

    env = BoatNaviEnv(render_mode="human", observation_type='pixel')

    observation = env.reset()

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

            print(observation, end='\r')
            #exit()
            #print(action)

    else:
        for _ in range(1000):
            observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

            if terminated or truncated:
                observation, info = env.reset()

    env.close()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/