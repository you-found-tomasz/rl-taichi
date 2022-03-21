import gym
from gym.utils import seeding
import numpy as np
from particle_simulator_wrapper import Particle_Simulator


class Taichi_v0 (gym.Env):
    # possible actions
    MOVE_LF = 0
    MOVE_RT = 1

    # possible positions
    LF_MIN = 1
    RT_MAX = 10

    # possible rewards
    REWARD_AWAY = -2
    REWARD_STEP = -1

    metadata = {
        "render.modes": ["human"]
        }


    def __init__ (self):
        # the action space ranges [0, 1] where:
        #  `0` move left
        #  `1` move right
        self.simulator = Particle_Simulator()
        self.RT_MAX = int(self.simulator.first_quarter_index.shape[0])
        self.action_space = gym.spaces.Discrete(self.RT_MAX)

        # NB: Ray throws exceptions for any `0` value Discrete
        # observations so we'll make position a 1's based value
        #self.observation_space = gym.spaces.Discrete(self.RT_MAX)
        self.observation_space = gym.spaces.MultiBinary(self.RT_MAX)
        #self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.RT_MAX, 1), dtype=np.int)
        #self.observation_space = gym.spaces.Tuple((gym.spaces.Discrete(self.RT_MAX), gym.spaces.MultiBinary(self.RT_MAX, 1)))

        # possible positions to chose on `reset()`
        self.goal = int((self.LF_MIN + self.RT_MAX - 1) / 2)
        self.seed()
        self.reset()


    def reset (self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.count = 0
        self.state = np.ones(self.RT_MAX)

        # for this environment, state is simply the position
        self.reward = 0
        self.done = False
        self.info = {}
        print("reset")
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
            self.simulator.cloth_broken = False
            print("cloth broken")
        else:
            assert self.action_space.contains(action)
            self.count += 1

            self.state[action] = 0
            if self.count %75 == 0:
                self.simulator.simulate(self.state)
                if self.simulator.cloth_broken == False:
                    self.reward = int(self.state.shape[0] - self.state.sum())
            self.info["dist"] = self.goal
            self.info["cloth"] = self.simulator.cloth_broken

        try:
            assert self.observation_space.contains(self.state)
        except AssertionError:
            print("INVALID STATE", self.state)

        return [self.state, self.reward, self.done, self.info]


    def render (self, mode="human"):
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
        #s = "position: {:2d}  reward: {:2d}  info: {}"
        #print(s.format(self.state, self.reward, self.info))
        print(self.state.sum(), self.reward)



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
