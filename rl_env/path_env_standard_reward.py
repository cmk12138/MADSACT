# -*- coding: utf-8 -*-
#开发者：Bright Fang
#开发时间：2023/7/20 23:30
import numpy as np
import copy
import gym
from assignment import constants as C
from gym import spaces
import math
import random
import pygame
from assignment.components import player
from assignment import tools
from assignment.components import info
class RlGame(gym.Env):
    def __init__(self, n,m,render=False):
        self.hero_num = n
        self.enemy_num = m
        self.obstacle_num=1
        self.goal_num=1
        self.Render=render
        self.game_info = {
            'epsoide': 0,
            'hero_win': 0,
            'enemy_win': 0,
            'win': '未知',
        }
        if self.Render:
            pygame.init()
            pygame.mixer.init()
            self.SCREEN = pygame.display.set_mode((C.SCREEN_W, C.SCREEN_H))

            pygame.display.set_caption("基于深度强化学习的无人机路径规划")

            self.GRAPHICS = tools.load_graphics('E:/PycharmProjects/code/backup/path planning/assignment/source/image')

            self.SOUND = tools.load_sound('E:/PycharmProjects/code/backup/path planning/assignment/source/music')
            self.clock = pygame.time.Clock()
            self.mouse_pos=(100,100)
            pygame.time.set_timer(C.CREATE_ENEMY_EVENT, C.ENEMY_MAKE_TIME)
            # self.res, init_extra, update_extra, skip_override, waypoints = simulate(filename='')

        # else:
        #     self.dispaly=None
        low = np.array([-1,-1])
        high=np.array([1,1])
        # self.action_space =spaces.Discrete(21)
        # self.action_space = spaces.Discrete(2)
        self.action_space=spaces.Box(low=low,high=high,dtype=np.float32)
        # self.action_space = [spaces.Discrete(2),spaces.Discrete(2),spaces.Discrete(2),spaces.Discrete(2),
        #                      spaces.Discrete(2),spaces.Discrete(2),spaces.Discrete(2),spaces.Discrete(2),
        #                      spaces.Discrete(2),spaces.Discrete(2),spaces.Discrete(2),spaces.Discrete(2),
        #                      spaces.Discrete(2),spaces.Discrete(2),spaces.Discrete(2),spaces.Discrete(2),
        #                      spaces.Discrete(2),spaces.Discrete(2),spaces.Discrete(2),spaces.Discrete(2),]
    
    def start(self):
        # self.game_info=game_info
        self.finished=False
        # self.next='game_over'
        self.set_battle_background()#战斗的背景
        self.set_enemy_image()
        self.set_hero_image()
        self.set_obstacle_image()
        self.set_goal_image()
        self.info = info.Info('battle_screen',self.game_info)
        # self.state = 'battle'
        self.counter_1 = 0
        self.counter_hero = 0
        self.enemy_counter=0
        self.enemy_counter_1 = 0
        #又定义了一个参数，为了放在start函数里重置
        self.enemy_num_start=self.enemy_num
        self.trajectory_x,self.trajectory_y=[],[]
        self.enemy_trajectory_x,self.enemy_trajectory_y=[[] for i in range(self.enemy_num)],[[] for i in range(self.enemy_num)]
        # RL状态
        # self.hero_state = np.zeros((self.hero_num, 4))
        # self.hero_α = np.zeros((self.hero_num, 1))
        self.uav_obs_check= np.zeros((self.hero_num, 1))

    def set_battle_background(self):
        self.battle_background = self.GRAPHICS['background']
        self.battle_background = pygame.transform.scale(self.battle_background,C.SCREEN_SIZE)  # 缩放
        self.view = self.SCREEN.get_rect()
        #若要移动的背景图像，请用下面的代码替换
        # bg1=player.BackgroundSprite(image_name='background3',size=C.SCREEN_SIZE)
        # bg2=player.BackgroundSprite(image_name='background3',size=C.SCREEN_SIZE)
        # bg2.rect.y=-bg2.rect.height
        # self.background_group=pygame.sprite.Group(bg1,bg2)

    def set_hero_image(self):
        self.hero = self.__dict__
        self.hero_group = pygame.sprite.Group()
        self.hero_image = self.GRAPHICS['fighter-blue']
        for i in range(self.hero_num):
            self.hero['hero'+str(i)]=player.Hero(image=self.hero_image)
            self.hero_group.add(self.hero['hero'+str(i)])

    def set_enemy_image(self):
        self.enemy = self.__dict__
        self.enemy_group = pygame.sprite.Group()
        self.enemy_image = self.GRAPHICS['fighter-green']
        for i in range(self.enemy_num):
            self.enemy['enemy'+str(i)]=player.Enemy(image=self.enemy_image)
            self.enemy_group.add(self.enemy['enemy'+str(i)])

    def set_hero(self):
        self.hero = self.__dict__
        self.hero_group = pygame.sprite.Group()
        for i in range(self.hero_num):
            self.hero['hero'+str(i)]=player.Hero()
            self.hero_group.add(self.hero['hero'+str(i)])

    def set_enemy(self):
        self.enemy = self.__dict__
        self.enemy_group = pygame.sprite.Group()
        for i in range(self.enemy_num):
            self.enemy['enemy'+str(i)]=player.Enemy()
            self.enemy_group.add(self.enemy['enemy'+str(i)])

    def set_obstacle_image(self):
        self.obstacle = self.__dict__
        self.obstacle_group = pygame.sprite.Group()
        self.obstacle_image = self.GRAPHICS['hole']
        for i in range(self.obstacle_num):
            self.obstacle['obstacle'+str(i)]=player.Obstacle(image=self.obstacle_image)
            self.obstacle_group.add(self.obstacle['obstacle'+str(i)])

    def set_obstacle(self):
        self.obstacle = self.__dict__
        self.obstacle_group = pygame.sprite.Group()
        for i in range(self.obstacle_num):
            self.obstacle['obstacle'+str(i)]=player.Obstacle()
            self.obstacle_group.add(self.obstacle['obstacle'+str(i)])

    def set_goal_image(self):
        self.goal = self.__dict__
        self.goal_group = pygame.sprite.Group()
        self.goal_image = self.GRAPHICS['goal']
        for i in range(self.goal_num):
            self.goal['goal'+str(i)]=player.Goal(image=self.goal_image)
            self.goal_group.add(self.goal['goal'+str(i)])

    def set_goal(self):
        self.goal = self.__dict__
        self.goal_group = pygame.sprite.Group()
        for i in range(self.goal_num):
            self.goal['goal'+str(i)]=player.Goal()
            self.goal_group.add(self.goal['goal'+str(i)])

    def update_game_info(self):#死亡后重置数据
        self.game_info['epsoide'] += 1
        self.game_info['enemy_win'] = self.game_info['epsoide'] - self.game_info['hero_win']

    def reset(self):#reset的仅是环境状态，
        # obs=np.zeros((self.n, 4))#这是个二维矩阵，n*2维,现在只考虑一个己方无人机，所以现在是一个一维的
        # game_info=self.my_game.state.game_info
        # self.my_game.state.start(game_info)
        if self.Render:
            self.start()
        else:
            self.set_hero()
            self.set_enemy()
            self.set_goal()
            self.set_obstacle()
        self.team_counter = 0
        self.done = False
        self.hero_state = np.zeros((self.hero_num+self.enemy_num,7))
        self.hero_α = np.zeros((self.hero_num, 1))
        
        # ChatGPT Modified Here: Added enemy state to observation
        self.prev_d_goal_leader = math.hypot(self.hero0.init_x - self.goal0.init_x,
                                     self.hero0.init_y - self.goal0.init_y)
        # ChatGPT Modified End
        
        return np.array([[self.hero0.init_x/1000,self.hero0.init_y/1000,self.hero0.speed/30,self.hero0.theta*57.3/360
                            ,self.goal0.init_x/1000, self.goal0.init_y/1000,0],
                         [self.enemy0.init_x / 1000, self.enemy0.init_y / 1000, self.enemy0.speed / 30,
                         self.enemy0.theta * 57.3 / 360
                            , self.hero0.init_x/1000, self.hero0.init_y/1000,self.hero0.speed / 30],#训练修改数组数量为2（要注释）——————————————————————————————————————————————————————————————————————测试修改为4
                        #  [self.enemy1.init_x / 1000, self.enemy1.init_y / 1000, self.enemy1.speed / 30,
                        #   self.enemy1.theta * 57.3 / 360
                        #      , self.hero0.init_x / 1000, self.hero0.init_y / 1000, self.hero0.speed / 30],
                        #  [self.enemy2.init_x / 1000, self.enemy2.init_y / 1000, self.enemy2.speed / 30,
                        #   self.enemy2.theta * 57.3 / 360
                        #      , self.hero0.init_x / 1000, self.hero0.init_y / 1000, self.hero0.speed / 30],
                        #   [self.enemy3.init_x / 1000, self.enemy3.init_y / 1000, self.enemy3.speed / 30,
                        #    self.enemy3.theta * 57.3 / 360
                        #       , self.hero0.init_x / 1000, self.hero0.init_y / 1000, self.hero0.speed / 30],
                         ])
        #np.array([self.my_game.state.hero['hero0'].posx/1000,self.my_game.state.hero['hero0'].posy/1000,self.my_game.state.hero['hero0'].speed/2,self.my_game.state.hero['hero0'].theta*57.3/360])#np.zeros((self.n,2)).flatten()


    def step(self, action):
            """
            Standard multi-agent formation reward (leader-to-goal + followers maintain formation).
            action: np.ndarray/list shape (hero_num+enemy_num, 2), each in [-1, 1]
            """
            # ------------------------------------------------------------
            # 1) Apply actions to advance the environment (transition)
            # ------------------------------------------------------------
            # Leader(s)
            for i in range(self.hero_num):
                if not self.hero['hero' + str(i)].dead:
                    self.hero['hero' + str(i)].update(action[i], self.Render)
            # Followers
            for j in range(self.enemy_num):
                idx = self.hero_num + j
                if not self.enemy['enemy' + str(j)].dead:
                    self.enemy['enemy' + str(j)].update(action[idx], self.Render)

            # ------------------------------------------------------------
            # 2) Compute rewards on the *new* state
            # ------------------------------------------------------------
            r = np.zeros((self.hero_num + self.enemy_num, 1))
            self.done = False

            # Hyper-parameters (kept simple & smooth; adjust if needed)
            # Goal/progress
            w_goal_prog = 0.05         # scales (prev_d - d) in pixels to reward
            progress_clip = 40.0
            w_goal_dist = 0.2          # small shaping to keep gradient when progress ~0

            # Safety (obstacle / edge)
            obs_hit = 20.0
            obs_safe = 60.0
            w_obs_leader = 2.0
            w_obs_follower = 1.0

            edge_margin = 50.0
            w_edge = 1.0

            # Formation
            d_des = 30.0               # desired leader-follower distance (pixels)
            sigma_d = 10.0
            tol_d = 8.0                # for team_counter
            w_form_f = 0.6
            w_form_leader = 0.2

            # Velocity / heading matching
            sigma_v = 3.0
            w_vel = 0.3
            w_head = 0.1
            w_ahead = 0.3              # penalty when follower is "ahead" of leader along heading

            # Regularization
            w_act = 0.01               # action energy penalty (squared)
            w_time = 0.01              # per-step time penalty (encourage efficiency)

            # Team shaping (shared progress signal)
            w_team_prog = 0.02

            # Helper: smooth edge penalty
            def edge_penalty(x, y):
                d = min(x, C.SCREEN_W - x, y, C.SCREEN_H - y)
                if d >= edge_margin:
                    return 0.0
                t = (edge_margin - d) / edge_margin
                return -w_edge * (t * t)

            # Helper: smooth obstacle proximity penalty (0 outside safe radius)
            def obstacle_penalty(d, w):
                if d <= obs_hit:
                    return None  # indicate "hit"
                if d >= obs_safe:
                    return 0.0
                t = (obs_safe - d) / obs_safe
                return -w * (t * t)

            # ------------------------------------------------------------
            # Leader reward (assume hero0 is the leader)
            # ------------------------------------------------------------
            leader = self.hero0
            goal = self.goal0
            obstacle = self.obstacle0

            # Distances
            d_goal = math.hypot(leader.posx - goal.init_x, leader.posy - goal.init_y)
            d_obs = math.hypot(leader.posx - obstacle.init_x, leader.posy - obstacle.init_y)

            # Progress-based goal reward (potential difference)
            progress = float(np.clip(self.prev_d_goal_leader - d_goal, -progress_clip, progress_clip))
            r_goal = w_goal_prog * progress - w_goal_dist * (d_goal / 1000.0)

            # Obstacle / edge
            r_edge = edge_penalty(leader.posx, leader.posy)
            obs_term = obstacle_penalty(d_obs, w_obs_leader)
            o_flag = 0
            if obs_term is None and not leader.dead:
                # collision
                o_flag = 1
                r_obs = -500.0
                leader.die()
                self.done = True
            else:
                r_obs = 0.0 if obs_term is None else obs_term

            # Terminal goal
            if (d_goal < 40.0) and (not leader.dead):
                r_goal = 1000.0
                leader.win = True
                leader.die()
                self.done = True

            # Action penalty (leader action index 0)
            a0 = float(action[0][0])
            p0 = float(action[0][1])
            r_act_leader = -w_act * (a0 * a0 + p0 * p0)

            # Formation cohesion reward for leader (average over followers)
            form_scores = []
            # heading unit (pygame y-down, so forward is (cos, -sin))
            hx = math.cos(leader.theta)
            hy = -math.sin(leader.theta)

            for j in range(self.enemy_num):
                follower = self.enemy['enemy' + str(j)]
                rx = follower.posx - leader.posx
                ry = follower.posy - leader.posy
                d_lf = math.hypot(rx, ry)

                # distance score in [-1, 1]
                score_d = math.exp(-((d_lf - d_des) ** 2) / (2.0 * sigma_d * sigma_d))
                form_scores.append(2.0 * score_d - 1.0)

            r_form_leader = w_form_leader * (sum(form_scores) / len(form_scores)) if form_scores else 0.0

            # Shared team progress
            r_team = w_team_prog * progress

            # Total leader reward
            r[0] = r_goal + r_edge + r_obs + r_form_leader + r_team + r_act_leader - w_time

            # Update prev distance AFTER computing reward
            self.prev_d_goal_leader = d_goal

            # ------------------------------------------------------------
            # Follower rewards
            # ------------------------------------------------------------
            # Count good-formation steps (all followers within tolerance & behind leader)
            all_good = True

            for j in range(self.enemy_num):
                idx = self.hero_num + j
                follower = self.enemy['enemy' + str(j)]

                # Distances
                d_obs_f = math.hypot(follower.posx - obstacle.init_x, follower.posy - obstacle.init_y)
                rx = follower.posx - leader.posx
                ry = follower.posy - leader.posy
                d_lf = math.hypot(rx, ry)

                # Obstacle / edge (follower)
                r_edge_f = edge_penalty(follower.posx, follower.posy)
                obs_term_f = obstacle_penalty(d_obs_f, w_obs_follower)
                if obs_term_f is None and not follower.dead:
                    # follower hits obstacle: penalize but do NOT end episode (keeps original spirit)
                    r_obs_f = -200.0
                    follower.die()
                else:
                    r_obs_f = 0.0 if obs_term_f is None else obs_term_f

                # Formation distance reward (smooth, bounded)
                score_d = math.exp(-((d_lf - d_des) ** 2) / (2.0 * sigma_d * sigma_d))
                r_dist = w_form_f * (2.0 * score_d - 1.0)

                # Velocity matching (bounded)
                dv = abs(float(leader.speed) - float(follower.speed))
                score_v = math.exp(-(dv * dv) / (2.0 * sigma_v * sigma_v))
                r_vel = w_vel * (2.0 * score_v - 1.0)

                # Heading matching (bounded)
                dtheta = float(follower.theta - leader.theta)
                # wrap to [-pi, pi]
                while dtheta > math.pi:
                    dtheta -= 2.0 * math.pi
                while dtheta < -math.pi:
                    dtheta += 2.0 * math.pi
                score_h = (1.0 + math.cos(dtheta)) * 0.5
                r_head = w_head * (2.0 * score_h - 1.0)

                # "Behind leader" penalty: if follower is ahead along leader heading
                along = rx * hx + ry * hy
                r_ahead = -w_ahead * max(0.0, along / (d_des + 1e-6))

                # Too-close penalty (avoid collision with leader)
                d_min = 12.0
                r_sep = -1.0 * max(0.0, (d_min - d_lf) / d_min) ** 2

                # Action penalty
                a = float(action[idx][0])
                p = float(action[idx][1])
                r_act = -w_act * (a * a + p * p)

                # Shared team progress
                r_team = w_team_prog * progress

                # Total follower reward
                r[idx] = r_dist + r_vel + r_head + r_ahead + r_sep + r_edge_f + r_obs_f + r_team + r_act - w_time

                # formation quality check for team_counter
                behind_ok = (along <= 0.0)
                dist_ok = (abs(d_lf - d_des) <= tol_d)
                if not (behind_ok and dist_ok):
                    all_good = False

            if self.enemy_num > 0 and all_good and (not self.done):
                self.team_counter += 1

            # ------------------------------------------------------------
            # 3) Build observation/state (same structure as original)
            # ------------------------------------------------------------
            # Leader state
            self.hero_state[0] = [
                self.hero['hero0'].posx / 1000, self.hero['hero0'].posy / 1000,
                self.hero['hero0'].speed / 30,
                self.hero['hero0'].theta * 57.3 / 360,
                self.goal0.init_x / 1000, self.goal0.init_y / 1000,
                o_flag
            ]

            # Followers state(s)
            for j in range(self.enemy_num):
                idx = self.hero_num + j
                self.hero_state[idx] = [
                    self.enemy['enemy' + str(j)].posx / 1000, self.enemy['enemy' + str(j)].posy / 1000,
                    self.enemy['enemy' + str(j)].speed / 30,
                    self.enemy['enemy' + str(j)].theta * 57.3 / 360,
                    self.hero0.posx / 1000, self.hero0.posy / 1000, self.hero0.speed / 30
                ]

            hero_state = copy.deepcopy(self.hero_state)
            done = copy.deepcopy(self.done)

            # Keep return signature compatible with existing training code
            dis_1_agent_0_to_1 = 0.0
            if self.enemy_num > 0:
                dis_1_agent_0_to_1 = math.hypot(self.enemy0.posx - self.hero0.posx, self.enemy0.posy - self.hero0.posy)

            return hero_state, r, done, self.hero['hero0'].win, self.team_counter, dis_1_agent_0_to_1
    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                quit()
            elif event.type == pygame.MOUSEMOTION:
                self.mouse_pos = pygame.mouse.get_pos()
            elif event.type == C.CREATE_ENEMY_EVENT:
                C.ENEMY_FLAG = True
        # 画背景
        self.SCREEN.blit(self.battle_background, self.view)
        # 文字显示
        self.info.update(self.mouse_pos)
        # 画图
        self.draw(self.SCREEN)
        pygame.display.update()
        self.clock.tick(C.FPS)
    
    def draw(self,surface):
        # self.background_group.draw(surface)
        #敌占区的矩形
        pygame.draw.rect(surface, C.BLACK, C.ENEMY_AREA, 3)
        #目标星星
        # pygame.draw.polygon(surface, C.GREEN,[(200, 135), (205, 145), (215, 145), (210, 155), (213, 165), (200, 160), (187, 165), (190, 155), (185, 145), (195, 145)])
        pygame.draw.circle(surface, C.RED, (self.goal0.init_x, self.goal0.init_y), 1)
        pygame.draw.circle(surface, C.RED, (self.goal0.init_x, self.goal0.init_y), 40,1)
        # pygame.draw.circle(surface, C.GREEN, (self.goal0.init_x, self.goal0.init_y),100, 1)
        pygame.draw.circle(surface, C.BLACK, (self.obstacle0.init_x, self.obstacle0.init_y), 20, 1)
        # 画轨迹
        for i in range(1, len(self.trajectory_x)):
            pygame.draw.line(surface, C.BLUE, (self.trajectory_x[i - 1], self.trajectory_y[i - 1]), (self.trajectory_x[i], self.trajectory_y[i]))
        for j in range(self.enemy_num):
            for i in range(1, len(self.trajectory_x)):
                pygame.draw.line(surface, C.GREEN, (self.enemy_trajectory_x[j][i - 1], self.enemy_trajectory_y[j][i - 1]),
                                 (self.enemy_trajectory_x[j][i], self.enemy_trajectory_y[j][i]))
        #障碍物
        # pygame.draw.circle(surface, C.BLACK, (250, 300), 20)
        # 画自己
        self.hero_group.draw(surface)
        self.enemy_group.draw(surface)
        #障碍物
        self.obstacle_group.draw(surface)
        #self.obstacle_group.draw(surface)
        # 目标星星
        self.goal_group.draw(surface)
        #画文字信息
        self.info.draw(surface)
    # def close(self):
    #     pygame.display.quit()
    #     quit()
    def close(self):
        try:
            pygame.display.quit()
        except:
            pass

