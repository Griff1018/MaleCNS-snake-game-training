import numpy as np


def random_connectome_mask(n_neurons, sparsity=0.9, seed=0):
    """Stand-in for a real connectome adjacency matrix."""
    rng = np.random.default_rng(seed)

    mask = (
        rng.random((n_neurons, n_neurons)) > sparsity
    ).astype(np.float32)

    np.fill_diagonal(mask, 0)

    sign = rng.choice(
        [-1.0, 1.0],
        size=(n_neurons, n_neurons),
        p=[0.3, 0.7]
    )

    return mask, sign.astype(np.float32)


def load_biological_connectome(n_neurons=200, sparsity=0.9, seed=1):
    """Load biological fly connectome files if present.
    Falls back to a synthetic sparse connectome if missing.
    """
    try:
        mask = np.load("fly_mask.npy")
        sign = np.load("fly_sign.npy")

        expected_shape = (n_neurons, n_neurons)
        if mask.shape != expected_shape or sign.shape != expected_shape:
            raise ValueError(
                "Connectome dimensions are incompatible: "
                f"mask={mask.shape}, sign={sign.shape}, "
                f"expected={expected_shape}."
            )

        print("Loaded biological fly connectome.")

        return mask, sign

    except FileNotFoundError:
        print(
            "`fly_mask.npy` not found. "
            "Falling back to synthetic sparse connectome..."
        )

        return random_connectome_mask(
            n_neurons,
            sparsity=sparsity,
            seed=seed
        )


class ConnectomeBrain:
    """A leaky recurrent network whose connectivity is fixed by
    `mask` and `sign`.

    Trainable parameters:
        - Recurrent synapse strengths: self.W
        - Sensory input weights: self.W_in
    """

    def __init__(
        self,
        mask,
        sign,
        input_idx,
        output_idx,
        obs_dim,
        alpha=0.5,
        seed=0
    ):
        self.n = mask.shape[0]

        self.mask = mask
        self.sign = sign

        self.input_idx = np.array(input_idx, dtype=np.int64)
        self.output_idx = np.array(output_idx, dtype=np.int64)

        self.alpha = alpha
        self.obs_dim = obs_dim

        rng = np.random.default_rng(seed)

        # Trainable recurrent weights
        # Start below the recurrent saturation point.  The old 0..0.3 range
        # produced row sums of several units in the dense parts of the graph,
        # so tanh hid most of the useful sensory signal from the GA.
        self.W = (
            rng.random((self.n, self.n)).astype(np.float32) * 0.08
        ) * mask

        # Trainable sensory input weights
        #
        # Shape:
        #     number of input neurons × observation dimensions
        #
        # For Snake:
        #     input neurons × 25
        self.W_in = rng.normal(
            0,
            0.5,
            size=(len(input_idx), obs_dim)
        ).astype(np.float32)

        # Trainable sensory-to-action readout.  It gives evolution a short
        # path to useful turns while the fixed fly connectome remains the
        # recurrent processing core.
        self.W_out = np.zeros(
            (len(output_idx), obs_dim),
            dtype=np.float32
        )

        # Recurrent state
        self.state = np.zeros(
            self.n,
            dtype=np.float32
        )

    def reset(self):
        """Reset recurrent neural state."""
        self.state[:] = 0.0

    def _effective_W(self):
        """Apply fixed biological connectivity and synapse signs."""
        return self.W * self.mask * self.sign

    def step(self, obs):
        """Run one recurrent brain step.

        Parameters
        ----------
        obs : np.ndarray
            Observation vector with exactly `obs_dim` values.

        Returns
        -------
        np.ndarray
            Output logits for the action neurons.
        """

        obs = np.asarray(obs, dtype=np.float32)

        # Detect input-size mismatch immediately with a useful message.
        if obs.shape != (self.obs_dim,):
            raise ValueError(
                f"Brain observation size mismatch: "
                f"expected {self.obs_dim}, got {obs.shape[0]}"
            )

        drive = np.zeros(
            self.n,
            dtype=np.float32
        )

        # Sensory input → input neurons
        drive[self.input_idx] += self.W_in @ obs

        # Recurrent activation
        activation = np.tanh(self.state)

        recurrent = self._effective_W() @ activation

        # Leaky recurrent update
        self.state = (
            (1 - self.alpha) * self.state
            + self.alpha * (recurrent + drive)
        )

        # Action logits
        logits = self.state[self.output_idx] + self.W_out @ obs

        return logits

    def num_trainable_params(self):
        """Return number of trainable parameters."""
        return int(self.mask.sum()) + self.W_in.size + self.W_out.size


if __name__ == "__main__":
    # ============================================================
    # Snake configuration
    # ============================================================

    N_NEURONS = 200
    OBS_DIM = 25

    # 25 Snake observations are fed into the first 25 neurons.
    INPUT_IDX = range(0, 25)

    # 4 output neurons represent the 4 Snake actions.
    OUTPUT_IDX = range(25, 29)

    # ============================================================
    # Load connectome
    # ============================================================

    mask, sign = load_biological_connectome(
        n_neurons=N_NEURONS
    )

    # ============================================================
    # Create brain
    # ============================================================

    brain = ConnectomeBrain(
        mask=mask,
        sign=sign,
        input_idx=INPUT_IDX,
        output_idx=OUTPUT_IDX,
        obs_dim=OBS_DIM
    )

    # ============================================================
    # Test with a 25-dimensional observation
    # ============================================================

    dummy_obs = np.random.rand(
        OBS_DIM
    ).astype(np.float32)

    logits = brain.step(dummy_obs)

    print(
        "Neurons:",
        N_NEURONS,
        "| observation size:",
        OBS_DIM,
        "| trainable parameters:",
        brain.num_trainable_params()
    )

    print(
        "W_in shape:",
        brain.W_in.shape
    )

    print(
        "Action logits:",
        logits
    )
