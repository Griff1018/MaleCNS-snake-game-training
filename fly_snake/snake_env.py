import numpy as np

class SnakeEnv:
    def __init__(self, grid_size=12, max_steps_without_food=150):
        self.grid_size = grid_size
        self.max_steps_without_food = max_steps_without_food
        self.rng = np.random.default_rng()
        self.reset()

    def reset(self, seed=None):
        # A local RNG makes evaluation reproducible without changing the
        # genetic algorithm's mutation RNG.  This is important because every
        # candidate must face the same boards in a generation.
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        c = self.grid_size // 2
        self.snake = [(c, c), (c - 1, c), (c - 2, c)]
        self.direction = (1, 0)
        self.food = None
        self.food_count = 0
        self.steps = 0
        self.steps_since_food = 0
        self.completed = False
        self._place_food()
        return self._get_obs()

    def _place_food(self):
        occupied = set(self.snake)
        empty_cells = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in occupied
        ]

        if not empty_cells:
            self.food = None
            self.completed = True
            return

        self.food = empty_cells[self.rng.integers(len(empty_cells))]

    def _danger_in_direction(self, direction, distance):
        head_x, head_y = self.snake[0]
        dx, dy = direction

        x = head_x + dx * distance
        y = head_y + dy * distance

        # Wall
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return 1.0

        # Body
        if (x, y) in self.snake[:-1]:
            return 1.0

        return 0.0

    def _next_head_for_action(self, action):
        """Return the next head position for straight, left, or right."""
        dx, dy = self.direction
        if action == 1:
            dx, dy = -dy, dx
        elif action == 2:
            dx, dy = dy, -dx

        head_x, head_y = self.snake[0]
        return head_x + dx, head_y + dy

    def _direction_for_action(self, action):
        dx, dy = self.direction
        if action == 1:
            return -dy, dx
        if action == 2:
            return dy, -dx
        return dx, dy

    def _action_is_safe(self, action):
        x, y = self._next_head_for_action(action)
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return False

        # The current tail is safe: it moves away on a non-food step.
        return (x, y) not in self.snake[:-1]

    def _reachable_space_after(self, action):
        """Estimate how much free board remains after a safe action."""
        new_head = self._next_head_for_action(action)
        future_snake = [new_head] + self.snake[:]
        if new_head != self.food:
            future_snake.pop()

        blocked = set(future_snake)
        blocked.discard(new_head)
        seen = {new_head}
        queue = [new_head]

        for x, y in queue:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                candidate = (x + dx, y + dy)
                if (
                    0 <= candidate[0] < self.grid_size
                    and 0 <= candidate[1] < self.grid_size
                    and candidate not in blocked
                    and candidate not in seen
                ):
                    seen.add(candidate)
                    queue.append(candidate)

        return len(seen)

    def _forward_clearance_after(self, action):
        """Count safe forward cells after taking an action.

        This is deliberately conservative about the body and is used as an
        early-warning signal.  It makes the snake turn before it reaches the
        final square beside a wall.
        """
        x, y = self._next_head_for_action(action)
        dx, dy = self._direction_for_action(action)
        occupied = set(self.snake[:-1])
        clearance = 0

        while True:
            x += dx
            y += dy
            if (
                x < 0
                or x >= self.grid_size
                or y < 0
                or y >= self.grid_size
                or (x, y) in occupied
            ):
                return clearance
            clearance += 1

    def choose_action(self, logits):
        """Choose the learned action, with a tail-aware anti-trap guard.

        The guard only overrides the network when its preferred turn would
        enter a small pocket.  This prevents the long snake from tightening
        into a circle and colliding with itself at high scores.
        """
        logits = np.asarray(logits, dtype=np.float32)
        if logits.shape != (4,):
            raise ValueError(f"Expected four action logits, got {logits.shape}")

        # Action 3 is the legacy reverse/no-op output, so fold it into
        # straight rather than letting it distort the safety decision.
        action_scores = np.array(
            [max(logits[0], logits[3]), logits[1], logits[2]],
            dtype=np.float32
        )
        safe_actions = [
            action for action in range(3)
            if self._action_is_safe(action)
        ]
        if not safe_actions:
            return int(np.argmax(action_scores))

        preferred = max(safe_actions, key=lambda action: action_scores[action])

        # Do not wait for the wall to be the next square.  If the learned
        # action leaves at most one forward cell, take a safe turn with more
        # runway.  This keeps food-seeking from overpowering wall avoidance.
        clearance = {
            action: self._forward_clearance_after(action)
            for action in safe_actions
        }
        if clearance[preferred] <= 1:
            safer_turns = [
                action for action in safe_actions
                if clearance[action] >= 2
            ]
            if safer_turns:
                preferred = max(
                    safer_turns,
                    key=lambda action: (
                        clearance[action], action_scores[action]
                    )
                )

        # Flood-fill is only needed once the body is long enough to form
        # pockets.  Keeping it off early preserves fast training.
        if len(self.snake) < 12:
            return preferred

        space = {
            action: self._reachable_space_after(action)
            for action in safe_actions
        }
        largest_space = max(space.values())
        preferred_space = space[preferred]

        if (
            preferred_space < len(self.snake) + 8
            or preferred_space < largest_space * 0.60
        ):
            return max(
                safe_actions,
                key=lambda action: (space[action], action_scores[action])
            )

        return preferred

    def _get_obs(self):
        head_x, head_y = self.snake[0]
        direction = self.direction

        left_dir = (-direction[1], direction[0])
        right_dir = (direction[1], -direction[0])

        obs = []

        # Danger sensors: 1, 2, 3, 4, 5 cells
        # For each distance:
        # straight, left, right
        for dist in (1, 2, 3, 4, 5):
            obs.append(self._danger_in_direction(direction, dist))
            obs.append(self._danger_in_direction(left_dir, dist))
            obs.append(self._danger_in_direction(right_dir, dist))

        # Direction one-hot
        obs.extend([
            1.0 if direction == (0, -1) else 0.0,  # up
            1.0 if direction == (0, 1) else 0.0,   # down
            1.0 if direction == (-1, 0) else 0.0,  # left
            1.0 if direction == (1, 0) else 0.0    # right
        ])

        # Food direction
        if self.food is not None:
            fx, fy = self.food

            obs.extend([
                1.0 if fy < head_y else 0.0,  # food up
                1.0 if fy > head_y else 0.0,  # food down
                1.0 if fx < head_x else 0.0,  # food left
                1.0 if fx > head_x else 0.0   # food right
            ])

            # Normalized food position
            max_distance = max(1, self.grid_size - 1)

            obs.append((fx - head_x) / max_distance)
            obs.append((fy - head_y) / max_distance)

        else:
            obs.extend([
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0
            ])

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        # 0 = straight
        # 1 = left
        # 2 = right
        # 3 = reverse

        dx, dy = self.direction

        if action == 1:
            dx, dy = -dy, dx
        elif action == 2:
            dx, dy = dy, -dx
        elif action == 3:
            # Reverse is intentionally a no-op.  The action is retained for
            # compatibility with existing four-output brains, but a snake
            # must never be allowed to reverse into its own neck.
            pass

        new_direction = (dx, dy)
        head_x, head_y = self.snake[0]
        new_head = (head_x + dx, head_y + dy)

        self.steps += 1
        self.steps_since_food += 1

        # Collision detection
        wall_collision = (
            new_head[0] < 0
            or new_head[0] >= self.grid_size
            or new_head[1] < 0
            or new_head[1] >= self.grid_size
        )

        self_collision = new_head in self.snake[:-1]

        collision = wall_collision or self_collision

        if collision:
            if wall_collision:
                return (
                    self._get_obs(),
                    -1.0,
                    True,
                    {
                        "food_eaten": self.food_count,
                        "food_eaten_this_step": False,
                        "collision": True,
                        "wall_collision": True,
                        "self_collision": False,
                        "completed": False,
                        "timeout": False
                    }
                )

            return (
                self._get_obs(),
                -1.0,
                True,
                {
                    "food_eaten": self.food_count,
                    "food_eaten_this_step": False,
                    "collision": True,
                    "wall_collision": False,
                    "self_collision": True,
                    "completed": False,
                    "timeout": False
                }
            )

        self.direction = new_direction

        # Distance before moving
        old_distance = 0
        if self.food is not None:
            old_distance = abs(head_x - self.food[0]) + abs(head_y - self.food[1])

        # Move head
        self.snake.insert(0, new_head)

        food_eaten = False

        # Food eaten
        if self.food is not None and new_head == self.food:
            self.food_count += 1
            self.steps_since_food = 0
            food_eaten = True

            reward = 1.0

            self._place_food()

            if self.completed:
                reward += 10.0

            info = {
                "food_eaten": self.food_count,
                "food_eaten_this_step": True,
                "collision": False,
                "wall_collision": False,
                "self_collision": False,
                "completed": self.completed,
                "timeout": False
            }

            return self._get_obs(), reward, self.completed, info

        # Normal movement
        self.snake.pop()

        reward = 0.0

        # Distance shaping
        if self.food is not None:
            new_distance = (
                abs(new_head[0] - self.food[0])
                + abs(new_head[1] - self.food[1])
            )

            distance_change = old_distance - new_distance

            if distance_change > 0:
                reward += distance_change * 0.08
            elif distance_change < 0:
                reward += distance_change * 0.12

        # Time penalty
        time_penalty = 0.01 + 0.002 * self.steps_since_food
        reward -= time_penalty

        # Timeout
        timeout = self.steps_since_food >= self.max_steps_without_food

        if timeout:
            reward -= 0.5

        info = {
            "food_eaten": self.food_count,
            "food_eaten_this_step": food_eaten,
            "collision": False,
            "wall_collision": False,
            "self_collision": False,
            "completed": False,
            "timeout": timeout
        }

        return self._get_obs(), reward, timeout, info
