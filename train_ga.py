import json
import os
import pickle
import sys

import numpy as np

from fly_snake.snake_env import SnakeEnv
from fly_snake.brain import (
    ConnectomeBrain,
    load_biological_connectome
)


# ================================================================
# NETWORK
# ================================================================

N_NEURONS = 200

# 25 observations:
#
# 0-2    = danger distance 1
# 3-5    = danger distance 2
# 6-8    = danger distance 3
# 9-11   = danger distance 4
# 12-14  = danger distance 5
# 15-18  = current direction
# 19-22  = food direction
# 23-24  = normalized food dx/dy
N_INPUTS = 25

# 4 outputs:
#
# 0 = straight
# 1 = left
# 2 = right
# 3 = reverse
#
# Reverse is blocked by the environment.
N_OUTPUTS = 4

N_BRAIN_STEPS = 5


# ================================================================
# GENETIC ALGORITHM
# ================================================================

POP_SIZE = 100

# Each button press trains exactly 200 generations.
GENERATIONS_PER_RUN = 200

ELITE_COUNT = 10

MUTATION_RATE = 0.08
MUTATION_SCALE = 0.15

# The direct readout has only 100 parameters, so it can explore faster than
# the much larger recurrent connectome without destabilising it.
READOUT_MUTATION_RATE = 0.20
READOUT_MUTATION_SCALE = 0.25

EVAL_EPISODES = 4
MAX_STEPS = 500

# A fixed holdout set is used only to decide which model is saved for the
# dashboard.  Training boards rotate every generation.
VALIDATION_SEEDS = tuple(range(50000, 50008))


# ================================================================
# CONTINUOUS TRAINING
# ================================================================

CHECKPOINT_FILE = "ga_checkpoint.pkl"


# ================================================================
# FITNESS
# ================================================================

FOOD_REWARD = 3000.0

COMPLETION_BONUS = 200000.0

SELF_COLLISION_PENALTY = 500.0
WALL_COLLISION_PENALTY = 500.0

STEP_PENALTY = 0.05

# Small positive survival reward makes the objective prefer a safe route when
# two candidates have eaten the same amount of food.  Food remains dominant.
SURVIVAL_REWARD = 0.12

TIMEOUT_PENALTY = 500.0


# ================================================================
# BRAIN CREATION
# ================================================================

def create_brain(mask, sign, seed=0):
    return ConnectomeBrain(
        mask=mask,
        sign=sign,
        input_idx=range(
            0,
            N_INPUTS
        ),
        output_idx=range(
            N_INPUTS,
            N_INPUTS + N_OUTPUTS
        ),
        obs_dim=N_INPUTS,
        seed=seed
    )


# ================================================================
# EVALUATE ONE BRAIN
# ================================================================

def evaluate_brain(brain, env, episode_seeds):

    fitnesses = []
    food_results = []

    self_collision_count = 0
    wall_collision_count = 0
    completion_count = 0

    for episode, seed in enumerate(episode_seeds):

        # Common test boards remove most of the noise from fitness.  Without
        # this, a lucky food layout can make a weak brain look elite.
        obs = env.reset(seed=seed)
        brain.reset()

        done = False
        steps = 0

        food_eaten = 0
        previous_food = 0

        fitness = 0.0

        while not done and steps < MAX_STEPS:

            # ----------------------------------------------------
            # Let the recurrent brain settle before acting.
            # ----------------------------------------------------
            for _ in range(N_BRAIN_STEPS):
                logits = brain.step(obs)

            action = env.choose_action(logits)

            obs, reward, done, info = env.step(
                action
            )

            steps += 1

            # ----------------------------------------------------
            # Food tracking
            #
            # SnakeEnv should provide cumulative food_count
            # through info["food_eaten"].
            # ----------------------------------------------------
            food_eaten = info.get(
                "food_eaten",
                food_eaten
            )

            if food_eaten > previous_food:

                food_gained = (
                    food_eaten
                    - previous_food
                )

                fitness += (
                    food_gained
                    * FOOD_REWARD
                )

                previous_food = food_eaten

            # ----------------------------------------------------
            # General step cost.
            # ----------------------------------------------------
            fitness -= STEP_PENALTY
            fitness += SURVIVAL_REWARD

            # Keep environment reward:
            #
            # - distance shaping
            # - increasing time penalty
            # - basic environment food reward
            fitness += reward * 2.0

            # ----------------------------------------------------
            # Self collision
            # ----------------------------------------------------
            if info.get(
                "self_collision",
                False
            ):
                fitness -= (
                    SELF_COLLISION_PENALTY
                )

                self_collision_count += 1

            # ----------------------------------------------------
            # Wall collision
            # ----------------------------------------------------
            if info.get(
                "wall_collision",
                False
            ):
                fitness -= (
                    WALL_COLLISION_PENALTY
                )

                wall_collision_count += 1

            # ----------------------------------------------------
            # Timeout
            # ----------------------------------------------------
            if info.get(
                "timeout",
                False
            ):
                fitness -= TIMEOUT_PENALTY

            # ----------------------------------------------------
            # Completion
            # ----------------------------------------------------
            if info.get(
                "completed",
                False
            ):
                fitness += COMPLETION_BONUS

                completion_count += 1

        # --------------------------------------------------------
        # Hard maximum-step limit.
        # --------------------------------------------------------
        if not done and steps >= MAX_STEPS:
            fitness -= TIMEOUT_PENALTY

        fitnesses.append(fitness)
        food_results.append(food_eaten)

    mean_fitness = float(
        np.mean(fitnesses)
    )

    mean_food = float(np.mean(food_results))

    return (
        mean_fitness,
        mean_food,
        self_collision_count,
        wall_collision_count,
        completion_count
    )


# ================================================================
# MUTATION
# ================================================================

def mutate(brain):

    # ------------------------------------------------------------
    # Recurrent weights
    # ------------------------------------------------------------
    mutation_mask = (
        np.random.rand(
            *brain.W.shape
        ) < MUTATION_RATE
    ).astype(np.float32)

    noise = (
        np.random.randn(
            *brain.W.shape
        ).astype(np.float32)
        * MUTATION_SCALE
    )

    brain.W += (
        noise
        * mutation_mask
        * brain.mask
    )

    brain.W = np.abs(
        brain.W
    )

    brain.W = np.clip(
        brain.W,
        0.0,
        3.0
    )

    # Keep recurrent activity out of the saturated regime.  The biological
    # signs remain fixed in ConnectomeBrain; this only limits trainable
    # synapse magnitudes on rows with many incoming connections.
    stabilize_recurrent_weights(brain)

    # ------------------------------------------------------------
    # Input weights
    # ------------------------------------------------------------
    mutation_mask_in = (
        np.random.rand(
            *brain.W_in.shape
        ) < MUTATION_RATE
    ).astype(np.float32)

    noise_in = (
        np.random.randn(
            *brain.W_in.shape
        ).astype(np.float32)
        * MUTATION_SCALE
    )

    brain.W_in += (
        noise_in
        * mutation_mask_in
    )

    brain.W_in = np.clip(
        brain.W_in,
        -3.0,
        3.0
    )

    # Direct sensory-to-action weights are a compact, easy-to-evolve policy
    # layer.  The fly recurrent network remains active in every decision.
    mutation_mask_out = (
        np.random.rand(*brain.W_out.shape) < READOUT_MUTATION_RATE
    ).astype(np.float32)
    brain.W_out += (
        np.random.randn(*brain.W_out.shape).astype(np.float32)
        * READOUT_MUTATION_SCALE
        * mutation_mask_out
    )
    brain.W_out = np.clip(brain.W_out, -3.0, 3.0)


def stabilize_recurrent_weights(brain, max_row_strength=1.5):
    """Keep each recurrent neuron's incoming magnitude bounded."""
    row_strength = np.sum(
        np.abs(brain.W * brain.mask),
        axis=1,
        keepdims=True
    )
    brain.W *= np.minimum(
        1.0,
        max_row_strength / np.maximum(row_strength, 1e-6)
    )


# ================================================================
# PARENT SELECTION
# ================================================================

def select_parent(
    population,
    ranked,
    scores
):
    pool_size = min(
        20,
        len(ranked)
    )

    candidate_indices = (
        ranked[:pool_size]
    )

    candidate_scores = np.array(
        [
            scores[i]
            for i in candidate_indices
        ],
        dtype=np.float64
    )

    minimum = np.min(
        candidate_scores
    )

    weights = (
        candidate_scores
        - minimum
        + 1.0
    )

    if not np.isfinite(
        weights
    ).all():
        return np.random.choice(
            candidate_indices
        )

    if np.sum(weights) <= 0:
        return np.random.choice(
            candidate_indices
        )

    weights /= np.sum(
        weights
    )

    return np.random.choice(
        candidate_indices,
        p=weights
    )


# ================================================================
# SAVE BRAIN
# ================================================================

def save_brain(
    brain,
    prefix="trained"
):
    np.save(
        f"{prefix}_W.npy",
        brain.W
    )

    np.save(
        f"{prefix}_W_in.npy",
        brain.W_in
    )

    np.save(
        f"{prefix}_W_out.npy",
        brain.W_out
    )


# ================================================================
# TRAINING STATUS
# ================================================================

def write_status(
    status,
    generation,
    total_generations,
    best_fitness,
    best_food,
    best_self_collisions=0,
    best_wall_collisions=0
):
    data = {
        "status": status,
        "generation": generation,
        "total_generations": total_generations,
        "best_fitness": round(
            float(best_fitness),
            2
        ),
        "best_food": int(
            best_food
        ),
        "self_collisions": int(
            best_self_collisions
        ),
        "wall_collisions": int(
            best_wall_collisions
        )
    }

    with open(
        "training_status.json",
        "w"
    ) as f:
        json.dump(
            data,
            f
        )


# ================================================================
# SAVE GA CHECKPOINT
# ================================================================

def save_checkpoint(
    population,
    generation,
    best_ever_score,
    best_ever_food,
    best_ever_W,
    best_ever_W_in,
    best_ever_W_out
):
    checkpoint = {
        "generation": generation,

        "population_W": np.array([
            brain.W.copy()
            for brain in population
        ]),

        "population_W_in": np.array([
            brain.W_in.copy()
            for brain in population
        ]),

        "population_W_out": np.array([
            brain.W_out.copy()
            for brain in population
        ]),

        "best_ever_score": best_ever_score,

        "best_ever_food": best_ever_food,

        "best_ever_W": (
            None
            if best_ever_W is None
            else best_ever_W.copy()
        ),

        "best_ever_W_in": (
            None
            if best_ever_W_in is None
            else best_ever_W_in.copy()
        ),

        "best_ever_W_out": (
            None
            if best_ever_W_out is None
            else best_ever_W_out.copy()
        ),

        "numpy_random_state": np.random.get_state()
    }

    temp_file = (
        CHECKPOINT_FILE
        + ".tmp"
    )

    with open(
        temp_file,
        "wb"
    ) as f:
        pickle.dump(
            checkpoint,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    # Atomic replacement.
    os.replace(
        temp_file,
        CHECKPOINT_FILE
    )

    print(
        f"Checkpoint saved at generation "
        f"{generation}."
    )


# ================================================================
# LOAD GA CHECKPOINT
# ================================================================

def load_checkpoint(
    mask,
    sign
):
    if not os.path.exists(
        CHECKPOINT_FILE
    ):
        return None

    print(
        f"Loading GA checkpoint: "
        f"{CHECKPOINT_FILE}"
    )

    with open(
        CHECKPOINT_FILE,
        "rb"
    ) as f:
        checkpoint = pickle.load(f)

    population_W = checkpoint[
        "population_W"
    ]

    population_W_in = checkpoint[
        "population_W_in"
    ]

    population_W_out = checkpoint.get(
        "population_W_out"
    )

    if len(
        population_W
    ) != POP_SIZE:
        raise ValueError(
            "Checkpoint population size "
            f"({len(population_W)}) does not "
            f"match POP_SIZE ({POP_SIZE})."
        )

    expected_W = (
        N_NEURONS,
        N_NEURONS
    )

    expected_W_in = (
        N_INPUTS,
        N_INPUTS
    )

    if population_W.shape[1:] != expected_W:
        raise ValueError(
            "Checkpoint W dimensions are "
            f"{population_W.shape[1:]}, "
            f"expected {expected_W}."
        )

    if population_W_in.shape[1:] != expected_W_in:
        raise ValueError(
            "Checkpoint W_in dimensions are "
            f"{population_W_in.shape[1:]}, "
            f"expected {expected_W_in}."
        )

    expected_W_out = (N_OUTPUTS, N_INPUTS)
    if (
        population_W_out is not None
        and population_W_out.shape[1:] != expected_W_out
    ):
        raise ValueError(
            "Checkpoint W_out dimensions are "
            f"{population_W_out.shape[1:]}, "
            f"expected {expected_W_out}."
        )

    population = []

    for i in range(POP_SIZE):

        brain = create_brain(
            mask,
            sign,
            seed=i
        )

        brain.W = (
            population_W[i].copy()
        )

        brain.W_in = (
            population_W_in[i].copy()
        )

        if population_W_out is not None:
            brain.W_out = population_W_out[i].copy()

        # Checkpoints made by older versions can contain saturated recurrent
        # rows.  Normalize them when continuing so the improved evaluator is
        # not fed an already unstable population.
        stabilize_recurrent_weights(brain)

        population.append(
            brain
        )

    # Restore the RNG so continuation behaves
    # as if the previous process had never stopped.
    np.random.set_state(
        checkpoint[
            "numpy_random_state"
        ]
    )

    if population_W_out is None:
        print(
            "Legacy checkpoint detected; revalidating saved model "
            "with the new action readout."
        )
        best_ever_score = -float("inf")
        best_ever_food = 0.0
        best_ever_W = None
        best_ever_W_in = None
        best_ever_W_out = None
    else:
        best_ever_score = checkpoint["best_ever_score"]
        best_ever_food = checkpoint["best_ever_food"]
        best_ever_W = checkpoint["best_ever_W"]
        best_ever_W_in = checkpoint["best_ever_W_in"]
        best_ever_W_out = checkpoint.get("best_ever_W_out")

    return (
        population,
        checkpoint["generation"],
        best_ever_score,
        best_ever_food,
        best_ever_W,
        best_ever_W_in,
        best_ever_W_out
    )


# ================================================================
# MAIN
# ================================================================

def main(mode="restart"):

    if mode not in (
        "restart",
        "continue"
    ):
        raise ValueError(
            f"Invalid training mode: {mode}"
        )

    print(
        "Loading connectome..."
    )

    mask, sign = (
        load_biological_connectome(
            N_NEURONS,
            sparsity=0.9,
            seed=1
        )
    )

    print(
        f"Neurons: {N_NEURONS}"
    )

    print(
        f"Inputs: {N_INPUTS}"
    )

    print(
        f"Outputs: {N_OUTPUTS}"
    )

    print(
        f"Population: {POP_SIZE}"
    )

    print(
        f"Generations per run: "
        f"{GENERATIONS_PER_RUN}"
    )

    print(
        "Reward system:"
    )

    print(
        f"  Food reward: +{FOOD_REWARD}"
    )

    print(
        "  Distance shaping: 0.08 per cell"
    )

    print(
        "  Increasing time penalty:"
    )

    print(
        "    0.01 + 0.002 * steps_since_food"
    )

    print(
        f"  Self collision: "
        f"-{SELF_COLLISION_PENALTY}"
    )

    print(
        f"  Wall collision: "
        f"-{WALL_COLLISION_PENALTY}"
    )

    print(
        f"  Timeout: -{TIMEOUT_PENALTY}"
    )

    print()

    env = SnakeEnv(
        grid_size=12,
        max_steps_without_food=150
    )

    # ============================================================
    # INITIALIZE OR RESUME
    # ============================================================

    if mode == "continue":

        checkpoint = load_checkpoint(
            mask,
            sign
        )

        if checkpoint is None:
            raise FileNotFoundError(
                "No completed GA checkpoint "
                "was found. Run Restart GA first."
            )

        (
            population,
            start_generation,
            best_ever_score,
            best_ever_food,
            best_ever_W,
            best_ever_W_in,
            best_ever_W_out
        ) = checkpoint

        print()
        print(
            f"Continuing from generation "
            f"{start_generation}."
        )

        print(
            f"Previous best fitness: "
            f"{best_ever_score:.2f}"
        )

        print(
            f"Previous best food: "
            f"{best_ever_food}"
        )

    else:

        print(
            "Starting NEW GA training run."
        )

        population = [
            create_brain(
                mask,
                sign,
                seed=i
            )
            for i in range(POP_SIZE)
        ]

        start_generation = 0

        best_ever_score = -float(
            "inf"
        )

        best_ever_food = 0

        best_ever_W = None
        best_ever_W_in = None
        best_ever_W_out = None

    # This run will always add exactly 200
    # generations to the previous total.
    end_generation = (
        start_generation
        + GENERATIONS_PER_RUN
    )

    best_ever_brain = None

    if best_ever_W is not None:

        best_ever_brain = create_brain(
            mask,
            sign,
            seed=0
        )

        best_ever_brain.W = (
            best_ever_W.copy()
        )

        best_ever_brain.W_in = (
            best_ever_W_in.copy()
        )

        if best_ever_W_out is not None:
            best_ever_brain.W_out = best_ever_W_out.copy()

    write_status(
        "training",
        start_generation,
        end_generation,
        best_ever_score
        if np.isfinite(best_ever_score)
        else 0,
        best_ever_food
    )

    # ============================================================
    # GENERATIONS
    # ============================================================

    for gen in range(
        start_generation,
        end_generation
    ):

        scores = []
        foods = []

        self_collisions = []
        wall_collisions = []
        completions = []

        # --------------------------------------------------------
        # Evaluate population.
        # --------------------------------------------------------
        # A common board set per generation is fair, while changing the set
        # every generation stops the population memorising fixed layouts.
        episode_seeds = [
            10000 + gen * EVAL_EPISODES + episode
            for episode in range(EVAL_EPISODES)
        ]

        for brain in population:

            result = evaluate_brain(
                brain,
                env,
                episode_seeds
            )

            score = result[0]
            food = result[1]
            self_col = result[2]
            wall_col = result[3]
            completed = result[4]

            scores.append(score)
            foods.append(food)

            self_collisions.append(
                self_col
            )

            wall_collisions.append(
                wall_col
            )

            completions.append(
                completed
            )

        # --------------------------------------------------------
        # Rank population.
        # --------------------------------------------------------
        ranked = np.argsort(
            scores
        )[::-1]

        best_index = ranked[0]

        generation_best_score = float(
            scores[best_index]
        )

        generation_best_food = float(foods[best_index])

        generation_self_collisions = int(
            self_collisions[best_index]
        )

        generation_wall_collisions = int(
            wall_collisions[best_index]
        )

        generation_completions = int(
            completions[best_index]
        )

        # --------------------------------------------------------
        # Update the dashboard model using unseen, fixed holdout boards.
        # The training score is fair within a generation, but it is not
        # comparable across generations because each generation gets new
        # boards.  Validation makes the saved model generalize better.
        # --------------------------------------------------------
        validation_result = evaluate_brain(
            population[best_index],
            env,
            VALIDATION_SEEDS
        )
        validation_score = validation_result[0]
        validation_food = validation_result[1]

        if (
            validation_score
            > best_ever_score
        ):

            best_ever_score = (
                validation_score
            )

            best_ever_food = (
                validation_food
            )

            best_ever_W = (
                population[
                    best_index
                ].W.copy()
            )

            best_ever_W_in = (
                population[
                    best_index
                ].W_in.copy()
            )

            best_ever_W_out = (
                population[
                    best_index
                ].W_out.copy()
            )

            best_ever_brain = create_brain(
                mask,
                sign,
                seed=0
            )

            best_ever_brain.W = (
                best_ever_W.copy()
            )

            best_ever_brain.W_in = (
                best_ever_W_in.copy()
            )

            best_ever_brain.W_out = best_ever_W_out.copy()

            save_brain(
                best_ever_brain,
                "best_trained"
            )

            save_brain(
                best_ever_brain,
                "trained"
            )

        # --------------------------------------------------------
        # Training status.
        # --------------------------------------------------------
        write_status(
            "training",
            gen + 1,
            end_generation,
            best_ever_score,
            best_ever_food,
            generation_self_collisions,
            generation_wall_collisions
        )

        print(
            f"Gen {gen + 1:03d}/{end_generation} | "
            f"Best: {generation_best_score:8.1f} | "
            f"Mean Food: {generation_best_food:4.2f} | "
            f"Validation: {validation_food:4.2f} | "
            f"Self: {generation_self_collisions:3d} | "
            f"Wall: {generation_wall_collisions:3d} | "
            f"Complete: {generation_completions}"
        )

        # --------------------------------------------------------
        # Create next generation.
        # --------------------------------------------------------
        new_population = []

        # --------------------------------------------------------
        # Elitism.
        # --------------------------------------------------------
        for idx in ranked[
            :ELITE_COUNT
        ]:

            elite = create_brain(
                mask,
                sign
            )

            elite.W = (
                population[idx].W.copy()
            )

            elite.W_in = (
                population[idx].W_in.copy()
            )

            elite.W_out = population[idx].W_out.copy()

            new_population.append(
                elite
            )

        # --------------------------------------------------------
        # Mutated offspring.
        # --------------------------------------------------------
        while len(
            new_population
        ) < POP_SIZE:

            parent_idx = select_parent(
                population,
                ranked,
                scores
            )

            child = create_brain(
                mask,
                sign
            )

            child.W = (
                population[
                    parent_idx
                ].W.copy()
            )

            child.W_in = (
                population[
                    parent_idx
                ].W_in.copy()
            )

            child.W_out = population[parent_idx].W_out.copy()

            mutate(
                child
            )

            new_population.append(
                child
            )

        population = (
            new_population
        )

        # Make stopping safe: the next run can continue from the completed
        # generation instead of losing up to 199 generations of work.
        save_checkpoint(
            population=population,
            generation=gen + 1,
            best_ever_score=best_ever_score,
            best_ever_food=best_ever_food,
            best_ever_W=best_ever_W,
            best_ever_W_in=best_ever_W_in,
            best_ever_W_out=best_ever_W_out
        )

    # ============================================================
    # END OF 200-GENERATION RUN
    # ============================================================

    if best_ever_brain is not None:

        save_brain(
            best_ever_brain,
            "trained"
        )

    # ------------------------------------------------------------
    # Save a final checkpoint after the completed run.
    # ------------------------------------------------------------
    save_checkpoint(
        population=population,
        generation=end_generation,
        best_ever_score=best_ever_score,
        best_ever_food=best_ever_food,
        best_ever_W=best_ever_W,
        best_ever_W_in=best_ever_W_in,
        best_ever_W_out=best_ever_W_out
    )

    write_status(
        "complete",
        end_generation,
        end_generation,
        best_ever_score,
        best_ever_food
    )

    print()
    print(
        f"Training run finished at "
        f"generation {end_generation}!"
    )

    print(
        "Checkpoint saved."
    )

    print(
        "Use Continue GA to train "
        "the next 200 generations."
    )


# ================================================================
# COMMAND LINE
# ================================================================

if __name__ == "__main__":

    mode = "restart"

    if len(sys.argv) >= 2:
        mode = sys.argv[1].lower()

    if mode == "restart":

        # A restart deliberately discards the previous
        # evolutionary population.
        if os.path.exists(
            CHECKPOINT_FILE
        ):
            print(
                "Removing previous GA checkpoint..."
            )

            os.remove(
                CHECKPOINT_FILE
            )

    elif mode == "continue":

        if not os.path.exists(
            CHECKPOINT_FILE
        ):
            print(
                "ERROR: No GA checkpoint exists."
            )

            print(
                "Run Restart GA first."
            )

            sys.exit(1)

    else:

        print(
            "Usage:"
        )

        print(
            "  python train_ga.py restart"
        )

        print(
            "  python train_ga.py continue"
        )

        sys.exit(1)

    main(mode)
