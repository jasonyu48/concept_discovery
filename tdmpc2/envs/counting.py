import numpy as np
import gymnasium as gym
from PIL import Image, ImageDraw
from typing import List, Tuple, Optional


class CountingObjectsEnv(gym.Env):
    """Simple object-counting environment.

    Observation   : 64×64 RGB image (channels-first when returned).
    Action space  :
        - Continuous mode (default): 1-D Box in [-1,1].  Value
              < –τ (threshold) ⇒ remove one object (count –=1)
              >  τ (threshold) ⇒ add one object    (count +=1)
              else no-op.
        - Discrete mode (when enabled): 3-D one-hot vector [remove, no-op, add].
    Episode ends  : when current count == target_n  OR  step == max_steps.
    Reward        : 1.0 on successful termination, else 0.0.

    The colour & shape of objects are fixed per episode; positions are re-sampled every step.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    def __init__(self,
                 target_n: int = 3,
                 max_objects: int = 10,
                 img_size: int = 64,
                 max_steps: int = 10,
                 seed: Optional[int] = None,
                 threshold: float = 0.3333,
                 overlap_protection: bool = True,
                 discrete_action: bool = False):
        super().__init__()
        self.target_n = int(target_n)
        self.max_objects = int(max_objects)
        assert target_n >= 0 and target_n <= self.max_objects, f"target_n must be between 0 and max_objects"
        self.img_size = int(img_size)
        self.max_steps = int(max_steps)
        self.overlap_protection = bool(overlap_protection)
        self.discrete_action = bool(discrete_action)
        self.threshold = float(threshold)
        if not self.discrete_action:
            print(f"threshold: {self.threshold}")

        # Action space definition
        if self.discrete_action:
            # One-hot 3 actions: remove, no-op, add
            self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        else:
            # Continuous 1-D action in [-1,1]
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # Observation: C,H,W with C=3, values 0-255 uint8
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                 shape=(3, self.img_size, self.img_size),
                                                 dtype=np.uint8)

        self._rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    # ------------------------------------------------------------------
    # Core environment logic
    # ------------------------------------------------------------------
    def _generate_image(self):
        """Render current state to an RGB ndarray (C,H,W)."""
        img = Image.new('RGB', (self.img_size, self.img_size), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        r = 3  # base size parameter

        centres: List[Tuple[int, int]] = []

        def _sample_pos():
            """Sample a centre that does not overlap existing ones."""
            for _ in range(50):  # attempt limit
                cx_try = int(self._rng.integers(r, self.img_size - r))
                cy_try = int(self._rng.integers(r, self.img_size - r))
                if not self.overlap_protection:
                    return cx_try, cy_try
                ok = True
                for (px, py) in centres:
                    if (cx_try - px) ** 2 + (cy_try - py) ** 2 < (2 * r + 1) ** 2:
                        ok = False
                        break
                if ok:
                    return cx_try, cy_try
            # fallback: allow overlap if cannot find spot
            return cx_try, cy_try

        for _ in range(self.count):
            cx, cy = _sample_pos()
            centres.append((cx, cy))

            if self.shape == "circle":
                draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=self.color)
            elif self.shape == "square":
                draw.rectangle((cx - r, cy - r, cx + r, cy + r), fill=self.color)
            elif self.shape == "triangle":
                pts = [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)]
                draw.polygon(pts, fill=self.color)
            elif self.shape == "line":
                draw.line((cx - r, cy - r, cx + r, cy + r), fill=self.color, width=2)
            else:
                draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=self.color)  # fallback
        obs = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)  # C,H,W
        return obs

    # Gymnasium API -----------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.step_idx = 0
        # Start from a random count (could also start from 0)
        self.count = int(self._rng.integers(0, self.max_objects + 1))

        # Choose shape & colour for this episode ---------------------------------
        self.shape = self._rng.choice(["circle", "square", "triangle", "line"])
        self.color = tuple(int(x) for x in self._rng.integers(30, 226, size=3))
        obs = self._generate_image()
        self._last_obs = obs  # cache for true no-op reuse
        info = {}
        return obs, info

    def step(self, action):
        # Support both continuous scalar action and discrete one-hot action vector
        if self.discrete_action:
            # Expect a length-3 vector; accept any real-valued vector and use argmax
            a = np.asarray(action, dtype=np.float32).reshape(-1)
            if a.size != 3:
                raise ValueError(f"Discrete action mode expects vector of length 3, got shape {a.shape}")
            idx = int(np.argmax(a))  # 0: remove, 1: noop, 2: add
            delta = -1 if idx == 0 else (1 if idx == 2 else 0)
        else:
            # Continuous 1-D action in [-1,1]
            a = float(action[0])
            if a < -self.threshold:
                delta = -1
            elif a > self.threshold:
                delta = 1
            else:
                delta = 0
        # Update count within bounds; cannot go below 0 or above max_objects
        self.count = int(np.clip(self.count + delta, 0, self.max_objects))
        self.step_idx += 1

        terminated = bool(self.count == self.target_n)
        truncated = bool(self.step_idx >= self.max_steps)
        done = terminated or truncated

        reward = 1.0 if terminated else 0.0

        info = {
            'success': terminated,
            'terminated': terminated,
        }

        # If count changed, redraw; otherwise reuse previous frame for a true no-op
        if delta != 0:
            obs = self._generate_image()
            self._last_obs = obs
        else:
            # Return cached observation to keep pixels unchanged
            obs = self._last_obs
        return obs, reward, done, info

    def rand_act(self):
        """Environment-native random action sampler.
        In discrete mode, returns one-hot with lower probability of no-op.
        In continuous mode, returns uniform scalar in [-1, 1].
        """
        if self.discrete_action:
            idx = int(self._rng.choice(3, p=[0.4, 0.2, 0.4]))
            return np.eye(3, dtype=np.float32)[idx]
        return np.array([self._rng.uniform(-1.0, 1.0)], dtype=np.float32)

    # ------------------------------------------------------------------
    def render(self, mode='rgb_array'):
        if mode != 'rgb_array':
            raise NotImplementedError
        img = self._generate_image().transpose(1, 2, 0)  # H,W,C
        return img

    def close(self):
        pass


# ----------------------------------------------------------------------
# Factory used by tdmpc2.envs.__init__.py
# ----------------------------------------------------------------------


# ----------------- thin wrapper to drop Gymnasium info tuple -----------------


class CountingWrapper(gym.Wrapper):
    """Converts 5-tuple Gymnasium API to 4-tuple expected by TensorWrapper."""

    def __init__(self, env, cfg):
        super().__init__(env)
        self.cfg = cfg

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def step(self, action):
        # Underlying env already returns 4-tuple, keep behaviour
        return self.env.step(action)

    def rand_act(self):
        # Forward to underlying environment's random action sampler
        if hasattr(self.env, 'rand_act'):
            return self.env.rand_act()
        return self.env.action_space.sample()


# Update factory to include wrapper

def make_env(cfg):  # noqa: F811 – redefine to include wrapper
    if not isinstance(cfg.task, str) or not cfg.task.startswith('counting'):
        raise ValueError('Unknown counting task')
    try:
        target_n = int(cfg.task[len('counting'):])
    except ValueError:
        raise ValueError('Task name must be counting<number>, e.g. counting3')

    max_objects = getattr(cfg, 'max_objects', max(10, target_n * 2))
    env = CountingObjectsEnv(target_n=target_n,
                             max_objects=max_objects,
                             img_size=64,
                             max_steps=getattr(cfg, 'episode_length', 10),
                             discrete_action=bool(getattr(cfg, 'discrete_action', False)))
    env = CountingWrapper(env, cfg)
    env.max_episode_steps = env.env.max_steps  # unwrap level property
    return env 