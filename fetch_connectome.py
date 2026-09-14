"""Download a 200-neuron subgraph from the real male Drosophila CNS.

The matrix convention is ``matrix[target, source]`` because ConnectomeBrain
computes ``matrix @ activation``.  Do not transpose it.
"""

import json
import os
import sys

import numpy as np
from neuprint import Client, NeuronCriteria, fetch_adjacencies, fetch_neurons


SERVER = "https://neuprint.janelia.org"
# Full, synapse-level connectome of an adult male Drosophila CNS.
DATASET = "male-cns:v1.0"
N_NEURONS = 200
MIN_PRE = 100


def neurotransmitter_signs(neurons, body_ids):
    """Use GABA annotations where the dataset supplies them.

    Connectivity is measured EM data. Unannotated cells remain +1 instead of
    inventing a fake inhibitory pattern.
    """
    candidates = ("predictedNt", "nt", "neurotransmitter")
    nt_column = next((name for name in candidates if name in neurons), None)
    if nt_column is None:
        print("No transmitter annotation returned; using +1 polarity.")
        return {body_id: 1.0 for body_id in body_ids}

    transmitter_by_id = neurons.set_index("bodyId")[nt_column].to_dict()
    signs = {}
    for body_id in body_ids:
        transmitter = str(transmitter_by_id.get(body_id, "")).lower()
        signs[body_id] = -1.0 if "gaba" in transmitter else 1.0
    return signs


def main():
    token = os.environ.get("NEUPRINT_TOKEN")
    if not token:
        sys.exit(
            "NEUPRINT_TOKEN is not set. In PowerShell run: "
            '$env:NEUPRINT_TOKEN = "your-new-neuprint-token"'
        )

    print(f"Connecting to {SERVER} using dataset {DATASET}...")
    client = Client(SERVER, dataset=DATASET, token=token)
    print(f"Connected. Dataset: {client.dataset}")

    # API row order is not a ranking: select the highest-pre neurons
    # explicitly, from the real MaleCNS dataset.
    neurons, _ = fetch_neurons(
        NeuronCriteria(min_pre=MIN_PRE),
        returned_columns="all",
        client=client,
    )
    if len(neurons) < N_NEURONS:
        raise RuntimeError(
            f"Only {len(neurons)} neurons matched min_pre={MIN_PRE}; "
            f"need {N_NEURONS}."
        )

    selected = (
        neurons.sort_values(["pre", "post"], ascending=False)
        .head(N_NEURONS)
        .copy()
    )
    body_ids = selected["bodyId"].astype(np.int64).tolist()
    print(f"Selected {len(body_ids)} real male-CNS neurons.")

    # The second return value is a directed body-pair table when omit_rois is
    # enabled; the old script accidentally treated the two-item return value
    # as a single DataFrame.
    _, connections = fetch_adjacencies(
        sources=body_ids,
        targets=body_ids,
        min_total_weight=1,
        omit_rois=True,
        weight_props=["weight"],
        client=client,
    )

    index = {body_id: i for i, body_id in enumerate(body_ids)}
    source_sign = neurotransmitter_signs(selected, body_ids)
    mask = np.zeros((N_NEURONS, N_NEURONS), dtype=np.float32)
    sign = np.ones((N_NEURONS, N_NEURONS), dtype=np.float32)

    for row in connections.itertuples(index=False):
        source = int(row.bodyId_pre)
        target = int(row.bodyId_post)
        if source == target:
            continue

        # target, source matches ConnectomeBrain._effective_W() @ state.
        target_i = index[target]
        source_i = index[source]
        mask[target_i, source_i] = 1.0
        sign[target_i, source_i] = source_sign[source]

    np.fill_diagonal(mask, 0.0)
    metadata = {
        "source": "NeuPrint",
        "server": SERVER,
        "dataset": DATASET,
        "neurons": N_NEURONS,
        "min_pre": MIN_PRE,
        "matrix_orientation": "target_by_source",
        "body_ids": body_ids,
        "synapses_in_subgraph": int(mask.sum()),
        "inhibitory_sign_rule": "GABA-labelled source neurons only",
    }

    np.save("fly_mask.npy", mask)
    np.save("fly_sign.npy", sign)
    with open("fly_connectome_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"Saved real male-CNS subgraph: {int(mask.sum())} directed "
        f"connections, {100 * (1 - mask.mean()):.1f}% sparse."
    )
    print("Saved fly_mask.npy, fly_sign.npy, and fly_connectome_metadata.json.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[Error] {exc}")
        sys.exit(1)
