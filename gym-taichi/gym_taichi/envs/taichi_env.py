import gym
from gym.utils import seeding
import numpy as np
from particle_simulator_wrapper import Particle_Simulator
import matplotlib.pyplot as plt
import time

class Taichi_v0 (gym.Env):

    STEPS_UNTIL_SIMULATION = 200

    metadata = {
        "render.modes": ["human"]
        }

    def __init__ (self):

        self.simulator = Particle_Simulator()
        self.RT_MAX = int(self.simulator.first_quarter.shape[0])
        self.action_space = gym.spaces.Discrete(2)
        #self.observation_space = gym.spaces.Box(low=0, high=self.RT_MAX, shape=(1,), dtype=np.int)
        self.index_max = self.simulator.counter_max
        self.seed()
        self.reset()

        self.x_coordinate = 1000
        self.y_coordinate = 1000
        self.low = np.array([-self.x_coordinate, -self.y_coordinate], dtype=np.float64)
        self.high = np.array([self.x_coordinate, self.y_coordinate], dtype=np.float64)
        self.observation_space = gym.spaces.Box(self.low, self.high, dtype=np.float64)

    def reset (self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        #self.state = np.array([0])
        #self.state = (position, velocity)

        # for this environment, state is simply the position
        self.reward = 0
        self.done = False
        self.info = {}
        self.previous_state = self.index_max
        #self.previous_state_full = self.state
        self.previous_state_full = self.index_max
        self.index = 0
        print("reset")
        self.state = np.ndarray(shape=(2,), buffer=np.array([1, 1]))
        return self.state


    def step (self, action):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : Discrete

        Returns
        -------
        observation, reward, done, info : tuple
            observation (object) :
                an environment-specific object representing your observation of
                the environment.

            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.

            done (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)

            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        if self.done:
            # code should never reach this point
            print("EPISODE DONE!!!")

        elif self.simulator.cloth_broken == True:
            self.done = True;
            print("cloth broken")
            self.simulator.cloth_broken = False
        else:
            assert self.action_space.contains(action)
            if self.index == self.index_max -1:
                self.done = True
                self.index = 0
            else:
                self.index += 1
            self.state = self.simulator.update(self.index, action)

            if self.index % self.STEPS_UNTIL_SIMULATION == 0:
                self.simulator.simulate()
                if self.simulator.cloth_broken == False:
                    self.reward = int(self.previous_state - self.simulator.particle_indices.sum())
                    self.previous_state = self.simulator.particle_indices.sum()
            #else:
                #self.reward = 0
            #self.info["dist"] = self.goal
            self.info["cloth"] = self.simulator.cloth_broken
            #self.previous_state_full = self.state

        try:
            assert self.observation_space.contains(self.state)
        except AssertionError:
            print("INVALID STATE", self.state)

        #print(self.index, self.reward, self.simulator.particle_indices.sum())
        return [self.state, self.reward, self.done, self.info]


    def render (self, action, mode="human"):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with
        """
        first_quarter = self.simulator.X[self.simulator.first_quarter_index,:]
        first_quarter_reduced = first_quarter[np.where(self.state)[0],:]
        second_quarter_reduced = -first_quarter_reduced + self.simulator.center + self.simulator.center
        third_quarter_reduced = np.array([first_quarter_reduced[:,0], -first_quarter_reduced[:,1] + self.simulator.center[1] + self.simulator.center[1]]).T
        forth_quarter_reduced = np.array([-first_quarter_reduced[:,0] + self.simulator.center[1] + self.simulator.center[1], first_quarter_reduced[:,1]]).T
        plt.scatter(first_quarter_reduced[:,0], first_quarter_reduced[:,1])
        plt.scatter(second_quarter_reduced[:,0], second_quarter_reduced[:,1])
        plt.scatter(third_quarter_reduced[:,0], third_quarter_reduced[:,1])
        plt.scatter(forth_quarter_reduced[:,0], forth_quarter_reduced[:,1])
        plt.savefig("results_meshes/mesh_{}.png".format(time.time()), dpi=150)
        plt.clf()

        #s = "position: {:2d}  reward: {:2d}  info: {}"
        #print(s.format(self.state, self.reward, self.info))
        #self.simulator.simulate(self.state)
        #print(self.state.sum(), self.reward)



    def seed (self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def close (self):
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass
