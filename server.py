import asyncio
import json
import os
import subprocess
import sys
import traceback

import numpy as np

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import FileResponse, JSONResponse

from fly_snake.brain import (
    ConnectomeBrain,
    load_biological_connectome
)

from fly_snake.snake_env import SnakeEnv


app = FastAPI()


# ================================================================
# CONFIGURATION
# ================================================================

N_NEURONS = 200
N_INPUTS = 25
N_OUTPUTS = 4

TRAINING_GENERATIONS = 200

CHECKPOINT_FILE = "ga_checkpoint.pkl"


# ================================================================
# GLOBAL STATE
# ================================================================

high_score = 0

training_process = None


# ================================================================
# LOAD TRAINED BRAIN
# ================================================================

def get_brain():

    mask, sign = load_biological_connectome(
        N_NEURONS,
        sparsity=0.9,
        seed=1
    )

    brain = ConnectomeBrain(
        mask,
        sign,
        input_idx=range(
            0,
            N_INPUTS
        ),
        output_idx=range(
            N_INPUTS,
            N_INPUTS + N_OUTPUTS
        ),
        obs_dim=N_INPUTS,
        seed=1
    )

    if (
        os.path.exists("trained_W.npy")
        and
        os.path.exists("trained_W_in.npy")
    ):

        try:

            W = np.load(
                "trained_W.npy"
            )

            W_in = np.load(
                "trained_W_in.npy"
            )

            w_out_path = "trained_W_out.npy"
            W_out = (
                np.load(w_out_path)
                if os.path.exists(w_out_path)
                else None
            )

            expected_W = (
                N_NEURONS,
                N_NEURONS
            )

            expected_W_in = (
                N_INPUTS,
                N_INPUTS
            )

            expected_W_out = (
                N_OUTPUTS,
                N_INPUTS
            )

            if (
                W.shape == expected_W
                and
                W_in.shape == expected_W_in
                and
                (
                    W_out is None
                    or W_out.shape == expected_W_out
                )
            ):

                brain.W = W
                brain.W_in = W_in

                if W_out is not None:
                    brain.W_out = W_out

                print(
                    "Loaded trained brain."
                )

            else:

                print(
                    "WARNING: trained model "
                    "dimensions are incompatible."
                )

                print(
                    f"W shape: {W.shape}, "
                    f"expected: {expected_W}"
                )

                print(
                    f"W_in shape: {W_in.shape}, "
                    f"expected: {expected_W_in}"
                )

                print(
                    "Using fresh brain."
                )

        except Exception as e:

            print(
                "WARNING: Could not load "
                "trained brain."
            )

            print(
                f"Reason: {e}"
            )

            print(
                "Using fresh brain."
            )

    return brain, mask


# ================================================================
# NEURON DISPLAY POSITIONS
# ================================================================

np.random.seed(42)

neuron_positions = np.random.uniform(
    30,
    370,
    size=(N_NEURONS, 2)
).tolist()


# ================================================================
# DASHBOARD
# ================================================================

@app.get("/")
async def get_dashboard():

    return FileResponse(
        "index.html"
    )


# ================================================================
# TRAINING STATUS
# ================================================================

@app.get("/api/status")
async def get_status():

    train_info = {
        "status": "idle",
        "generation": 0,
        "total_generations": TRAINING_GENERATIONS,
        "best_fitness": 0,
        "best_food": 0,
        "self_collisions": 0,
        "wall_collisions": 0
    }

    if os.path.exists(
        "training_status.json"
    ):

        try:

            with open(
                "training_status.json",
                "r"
            ) as f:

                train_info = json.load(f)

        except Exception:
            pass

    return JSONResponse(
        train_info
    )


# ================================================================
# START / RESTART / CONTINUE TRAINING
# ================================================================

@app.post("/api/start-training")
async def start_training(
    request: Request
):

    global training_process

    # ------------------------------------------------------------
    # Read requested mode.
    # ------------------------------------------------------------
    try:

        body = await request.json()

    except Exception:

        body = {}

    mode = body.get(
        "mode",
        "restart"
    )

    if mode not in (
        "restart",
        "continue"
    ):

        return JSONResponse(
            {
                "status": "error",
                "detail": (
                    "Invalid training mode. "
                    "Use 'restart' or 'continue'."
                )
            },
            status_code=400
        )

    # ------------------------------------------------------------
    # Stop currently running training.
    # ------------------------------------------------------------
    if (
        training_process is not None
        and
        training_process.poll() is None
    ):

        print(
            "Stopping existing GA "
            "training process..."
        )

        training_process.terminate()

        try:

            training_process.wait(
                timeout=5
            )

        except subprocess.TimeoutExpired:

            print(
                "Training process did not "
                "terminate. Killing it..."
            )

            training_process.kill()

            training_process.wait()

    # ------------------------------------------------------------
    # CONTINUE
    # ------------------------------------------------------------
    if mode == "continue":

        if not os.path.exists(
            CHECKPOINT_FILE
        ):

            return JSONResponse(
                {
                    "status": "error",
                    "detail": (
                        "No completed 200-generation "
                        "checkpoint exists. "
                        "Run Restart GA first."
                    )
                },
                status_code=400
            )

        print(
            "Starting CONTINUED GA training..."
        )

    # ------------------------------------------------------------
    # RESTART
    # ------------------------------------------------------------
    else:

        print(
            "Starting NEW GA training..."
        )

        if os.path.exists(
            CHECKPOINT_FILE
        ):

            print(
                "Removing old GA checkpoint..."
            )

            try:

                os.remove(
                    CHECKPOINT_FILE
                )

            except Exception as e:

                return JSONResponse(
                    {
                        "status": "error",
                        "detail": (
                            "Could not remove "
                            f"old checkpoint: {e}"
                        )
                    },
                    status_code=500
                )

    # ------------------------------------------------------------
    # Initial status.
    # ------------------------------------------------------------
    if mode == "restart":

        status = {
            "status": "training",
            "generation": 0,
            "total_generations": 200,
            "best_fitness": 0,
            "best_food": 0,
            "self_collisions": 0,
            "wall_collisions": 0
        }

    else:

        # Read the previous checkpoint generation
        # so the dashboard can immediately show
        # the correct lifetime generation.
        previous_generation = 0

        try:

            import pickle

            with open(
                CHECKPOINT_FILE,
                "rb"
            ) as f:

                checkpoint = pickle.load(f)

            previous_generation = int(
                checkpoint.get(
                    "generation",
                    0
                )
            )

            previous_score = float(
                checkpoint.get(
                    "best_ever_score",
                    0
                )
            )

            previous_food = int(
                checkpoint.get(
                    "best_ever_food",
                    0
                )
            )

        except Exception:

            previous_generation = 0
            previous_score = 0
            previous_food = 0

        status = {
            "status": "training",
            "generation": previous_generation,
            "total_generations": (
                previous_generation
                + TRAINING_GENERATIONS
            ),
            "best_fitness": previous_score,
            "best_food": previous_food,
            "self_collisions": 0,
            "wall_collisions": 0
        }

    with open(
        "training_status.json",
        "w"
    ) as f:

        json.dump(
            status,
            f
        )

    # ------------------------------------------------------------
    # Start train_ga.py.
    # ------------------------------------------------------------
    training_process = subprocess.Popen(
        [
            sys.executable,
            "train_ga.py",
            mode
        ]
    )

    print(
        f"Started GA training "
        f"({mode}). "
        f"PID: {training_process.pid}"
    )

    return JSONResponse({
        "status": "started",
        "mode": mode,
        "pid": training_process.pid
    })


# ================================================================
# WEBSOCKET
# ================================================================

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket
):

    global high_score

    await websocket.accept()

    print(
        "WebSocket client connected."
    )

    try:

        env = SnakeEnv(
            grid_size=12,
            max_steps_without_food=150
        )

        episode_count = 1

        while True:

            # ----------------------------------------------------
            # Reload latest trained model at the
            # beginning of every episode.
            # ----------------------------------------------------
            brain, mask = get_brain()

            obs = env.reset()

            brain.reset()

            done = False

            food_eaten = 0

            while not done:

                for _ in range(5):

                    logits = brain.step(
                        obs
                    )

                action = env.choose_action(logits)

                (
                    obs,
                    reward,
                    done,
                    info
                ) = env.step(
                    action
                )

                food_eaten = info.get(
                    "food_eaten",
                    food_eaten
                )

                if food_eaten > high_score:

                    high_score = (
                        food_eaten
                    )

                # ------------------------------------------------
                # Training status.
                # ------------------------------------------------
                train_info = {
                    "status": "idle",
                    "generation": 0,
                    "total_generations": (
                        TRAINING_GENERATIONS
                    ),
                    "best_fitness": 0,
                    "best_food": 0,
                    "self_collisions": 0,
                    "wall_collisions": 0
                }

                if os.path.exists(
                    "training_status.json"
                ):

                    try:

                        with open(
                            "training_status.json",
                            "r"
                        ) as f:

                            train_info = json.load(
                                f
                            )

                    except Exception as e:

                        print(
                            "Could not read "
                            "training_status.json:",
                            e
                        )

                # ------------------------------------------------
                # Brain activity.
                # ------------------------------------------------
                activations = np.tanh(
                    brain.state
                ).tolist()

                active_count = int(
                    np.sum(
                        np.abs(
                            activations
                        ) > 0.15
                    )
                )

                # ------------------------------------------------
                # Food.
                # ------------------------------------------------
                if env.food is not None:

                    food_data = [
                        int(env.food[0]),
                        int(env.food[1])
                    ]

                else:

                    food_data = None

                # ------------------------------------------------
                # WebSocket payload.
                # ------------------------------------------------
                payload = {

                    "snake": [
                        [
                            int(x),
                            int(y)
                        ]
                        for x, y in env.snake
                    ],

                    "food": food_data,

                    "grid_size": int(
                        env.grid_size
                    ),

                    "score": int(
                        food_eaten
                    ),

                    "high_score": int(
                        high_score
                    ),

                    "episode": int(
                        episode_count
                    ),

                    "brain_state": activations,

                    "neuron_positions":
                        neuron_positions,

                    "active_neurons":
                        active_count,

                    "mean_activity": round(
                        float(
                            np.mean(
                                np.abs(
                                    activations
                                )
                            )
                        ),
                        3
                    ),

                    "training":
                        train_info,

                    "completed": bool(
                        info.get(
                            "completed",
                            False
                        )
                    ),

                    "collision": bool(
                        info.get(
                            "collision",
                            False
                        )
                    ),

                    "self_collision": bool(
                        info.get(
                            "self_collision",
                            False
                        )
                    ),

                    "wall_collision": bool(
                        info.get(
                            "wall_collision",
                            False
                        )
                    )
                }

                await websocket.send_text(
                    json.dumps(
                        payload
                    )
                )

                await asyncio.sleep(
                    0.08
                )

            episode_count += 1

            await asyncio.sleep(
                0.2
            )

    except Exception as e:

        print()
        print(
            "========== WEBSOCKET ERROR =========="
        )

        print(
            type(e).__name__,
            ":",
            e
        )

        traceback.print_exc()

        print(
            "====================================="
        )

        print()

        try:

            await websocket.close()

        except Exception:
            pass


# ================================================================
# RUN SERVER
# ================================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
