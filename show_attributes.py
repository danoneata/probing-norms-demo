import pdb

from math import log2
from collections import Counter

import altair as alt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from matplotlib import pyplot as plt

from probing_norms.get_results import add_metric, COMPUTE_METRICS, FEATURE_NAMES
from probing_norms.predict import NORMS_LOADERS
from probing_norms.utils import multimap, read_json
from probing_norms.scripts.eval_norm_correlations import (
    get_supercategories,
    get_best_supercategory_matches,
)


norms_loader_type = "mcrae-x-things"
norms_loader = NORMS_LOADERS[norms_loader_type]()
attribute_to_concepts, _, attributes = norms_loader()

supercategory_to_concepts = get_supercategories()
concept_and_supercategory = [
    (concept, supercategory)
    for supercategory, concepts in supercategory_to_concepts.items()
    for concept in concepts
]
concept_to_supercategories = multimap(concept_and_supercategory)


def compute_entropy(attribute):
    concepts = attribute_to_concepts[attribute]
    supercategories = [
        s for c in concepts for s in concept_to_supercategories.get(c, [])
    ]
    counts = Counter(supercategories)
    total = sum(counts.values())
    entropy = -sum((count / total) * log2(count / total) for count in counts.values())
    return {
        "entropy": entropy,
        "counts": counts,
    }


def get_attribute_to_intersection_norm():
    best_match_supercat = get_best_supercategory_matches(norms_loader_type)
    return {k: v[1] for k, v in best_match_supercat.items()}


def load_data_attribute_diversity():
    attribute_to_intersection_norm = get_attribute_to_intersection_norm()

    data = [
        {
            "attribute": attribute,
            "intersection-norm": attribute_to_intersection_norm[attribute],
            **compute_entropy(attribute),
        }
        for attribute in attributes
        if attribute in attribute_to_intersection_norm
    ]
    data = sorted(data, key=lambda x: x["entropy"], reverse=True)

    def select_keys(datum):
        return {k: datum[k] for k in ["attribute", "entropy", "intersection-norm"]}

    return pd.DataFrame([select_keys(datum) for datum in data])


def load_data_attribute_scores(metric):
    classifier_type = "linear-probe"
    embeddings_level = "concept"
    splits_type = "repeated-k-fold"
    norms_type = "mcrae-x-things"
    # metric = "score-f1-selectivity"

    def load_result_features(model):
        path = "static/results/{}-{}-{}-{}-{}.json".format(
            classifier_type,
            embeddings_level,
            splits_type,
            model,
            norms_type,
        )
        return read_json(path)

    results = [
        r
        for m in LABELS
        if m in FEATURE_NAMES
        for r in load_result_features(m)
    ]
    df = pd.DataFrame(results)
    if metric in COMPUTE_METRICS:
        df = add_metric(df, metric, norms_type=norms_type)
    df = df.pivot(index="feature", columns="model", values=metric)
    df = df.reset_index()
    df = df.rename(columns={"feature": "attribute"})

    return df


def load_data(metric):
    df1 = load_data_attribute_diversity()
    df2 = load_data_attribute_scores(metric)
    df = pd.merge(df1, df2, on="attribute", how="inner")
    return df


LABELS = ["entropy", "intersection-norm", "swin-v2-ssl", "dino-v2", "clip"]
NAMES = {
    "entropy": "Entropy",
    "intersection-norm": "Intersection norm",
}

for label in LABELS:
    if label not in NAMES:
        NAMES[label] = FEATURE_NAMES[label]


def main():
    HELP = """
    The first two options refer to two metrics that measure how much different attributes cut across supercategories (entropy and intersection norm).
    The next options refer to the F₁ selectivity score obtained for the Swin-v2, DINO v2, CLIP (image) models.
    """

    METRICS = ["score-f1", "score-f1-selectivity", "score-f1-selectivity-norm"]

    with st.sidebar:
        xlabel = st.selectbox(
            "x label:",
            LABELS,
            index=LABELS.index("intersection-norm"),
            help=HELP,
        )
        ylabel = st.selectbox(
            "y label:",
            LABELS,
            index=LABELS.index("dino-v2"),
            help=HELP,
        )
        st.markdown("---")
        metric = st.selectbox("Metric:", METRICS, index=1)

    df = load_data(metric)

    point_selector = alt.selection_point("point_selection")
    chart = (
        alt.Chart(df)
        .mark_circle(size=60)
        .encode(
            x=alt.X(xlabel, title=NAMES[xlabel]),
            y=alt.Y(ylabel, title=NAMES[ylabel]),
            tooltip=["attribute", xlabel, ylabel],
            fillOpacity=alt.condition(point_selector, alt.value(1), alt.value(0.3)),
        )
        .add_params(point_selector)
        .properties(width=400, height=600)
    )

    st.markdown("Click on a point in the chart to see more information about that attribute. You can use the `shift` key to select more points.")
    event = st.altair_chart(
        chart,
        use_container_width=True,
        key="alt_chart",
        on_select=lambda: None,
    )

    # st.write(event)
    points = event["selection"]["point_selection"]

    def find_closest(df, point):
        distances = ((df[xlabel] - point[xlabel]) ** 2 + (df[ylabel] - point[ylabel]) ** 2) ** 0.5
        return distances.idxmin()

    def show_more_info(point):
        i = find_closest(df, point)

        attribute = df.iloc[i]["attribute"]
        concepts = attribute_to_concepts[attribute]
        num_concepts = len(concepts)

        st.markdown("### Attribute: `{}`".format(attribute))
        st.markdown("Number of concepts: {}".format(num_concepts))

        sc = [(s, c) for c, s in concept_and_supercategory if c in concepts]
        sc = multimap(sc)
        sc = ["- {} ({} concepts) → {}".format(s, len(sc[s]), ", ".join(sc[s])) for s in sorted(sc.keys())]
        missing = set(concepts) - set(c for c, _ in concept_and_supercategory)
        sc = sc + ["- {} ({} concepts) → {}".format("N/A", len(missing), ", ".join(missing))]

        st.markdown("#### Concepts per supercategory")
        st.markdown("\n".join(sc))

    if isinstance(points, list):
        for point in points:
            show_more_info(point)



if __name__ == "__main__":
    main()
