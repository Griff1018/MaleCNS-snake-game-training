# MaleCNS Snake Training

A CPU-friendly Snake experiment driven by a small recurrent network whose
connectivity is sampled from the real adult male *Drosophila melanogaster* CNS
NeuPrint dataset. The project includes a browser dashboard, a genetic
algorithm (GA) trainer, a NeuPrint downloader, and runtime safety guards for
walls and self-trapping.

> This is **not** a full simulated fruit-fly brain. The project selects 200
> real neurons and their measured directed connections, then evolves the
> network weights for the Snake task.

## What it does

- Downloads a 200-neuron directed subgraph from NeuPrint dataset
  `male-cns:v1.0`.
- Uses the connection graph as a fixed recurrent-network topology.
- Evolves recurrent, sensory, and sensory-to-action weights with a GA.
- Shows live Snake play, training status, and neuron activity at a local web
  dashboard.
- Saves an atomic checkpoint after each completed generation, so training can
  be continued after stopping.

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or newer
- A NeuPrint account and personal API token to download real connectome data

Create an isolated environment in PowerShell:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy fastapi "uvicorn[standard]" neuprint-python
```

If PowerShell blocks activation, use the environment interpreter directly:

```powershell
.\venv\Scripts\python.exe -m pip install numpy fastapi "uvicorn[standard]" neuprint-python
```

## Download the real male-CNS connectome

1. Create or sign in to a NeuPrint account at
   [neuprint.janelia.org](https://neuprint.janelia.org/).
2. Create a personal API token in your NeuPrint account.
3. In the same PowerShell session, set the token as an environment variable:

   ```powershell
   $env:NEUPRINT_TOKEN = "paste-your-new-token-here"
   ```

4. Download the subgraph:

   ```powershell
   python fetch_connectome.py
   ```

This creates these local-only files:

```text
fly_mask.npy
fly_sign.npy
fly_connectome_metadata.json
```

They are intentionally ignored by Git. Never paste an API token into
`fetch_connectome.py`, commit it, or upload it to GitHub. If a token is ever
exposed, revoke it in NeuPrint and create a new one.

The downloader selects the 200 neurons with the largest presynaptic counts,
then fetches their measured directed connections. Its matrix is stored as
`target × source`, matching the recurrent update in `ConnectomeBrain`.

## Start the dashboard

Run:

```powershell
python server.py
```

Then open [http://localhost:8000](http://localhost:8000) in a browser.

The page shows the current Snake episode and provides **Restart GA** and
**Continue GA** buttons.

## Train the network

### New training run

Use this after downloading a new connectome, or whenever you want a clean
population:

```powershell
python train_ga.py restart
```

This starts a 200-generation run and replaces the prior GA checkpoint. The
dashboard's **Restart GA** button does the same thing.

### Continue training

After at least one completed generation/checkpoint:

```powershell
python train_ga.py continue
```

The dashboard's **Continue GA** button does the same thing. A continuation
adds 200 generations to the stored checkpoint. The trainer writes
`ga_checkpoint.pkl` after every completed generation, so stopping loses at
most the generation that is currently running.

### Understanding the metrics

- **Best**: highest training fitness in the current generation.
- **Mean Food**: average food obtained across that candidate's evaluation
  boards, not its single luckiest episode.
- **Validation**: food performance on separate holdout boards; this is used
  when selecting the model saved for the dashboard.
- **Self / Wall**: collision counts across the evaluation episodes.

The live dashboard runs a fresh random episode, so its score can still differ
from a training average.

## Model design

The Snake observation has 25 values: local danger at multiple distances,
current direction, food direction, and normalized food offset. The brain has:

- 200 recurrent neurons constrained by `fly_mask.npy` and `fly_sign.npy`.
- 25 sensory input weights.
- Four action outputs: straight, left, right, and a legacy no-op/reverse slot.
- A compact direct sensory-to-action readout that helps the GA learn basic
  turns efficiently while the connectome remains the recurrent core.

The environment also applies a tail-aware safety layer. It rejects an
immediate collision, turns before wall clearance is exhausted, and uses a
reachable-space check for long snakes to reduce circular self-traps.

## Important: changing the connectome

Downloading a new `fly_mask.npy`/`fly_sign.npy` changes the network topology.
Existing `trained_*.npy` files and `ga_checkpoint.pkl` may have the same
shape, but they were evolved for the old topology. Download the connectome
first, then run:

```powershell
python train_ga.py restart
```

## GitHub

`.gitignore` is a text file containing rules; it is not a folder. It excludes
local environments, tokens, downloaded connectome files, checkpoints, and
generated model weights. Those files remain available to Python on your
computer but are not uploaded to GitHub.

Before committing, check what will be uploaded:

```powershell
git status
```

Only stage source files and documentation, for example:

```powershell
git add README.md .gitignore server.py train_ga.py fetch_connectome.py fly_snake index.html
```

## Project layout

```text
fetch_connectome.py  Download the real MaleCNS subgraph from NeuPrint
train_ga.py          Genetic-algorithm trainer and checkpoints
server.py             FastAPI dashboard server
index.html            Browser dashboard
fly_snake/brain.py    Connectome-constrained recurrent network
fly_snake/snake_env.py Snake environment and safety controller
```
