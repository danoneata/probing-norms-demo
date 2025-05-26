import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from toolz import partition_all

from probing_norms.data import load_binder_dense
from probing_norms.get_results import (
    FEATURE_NAMES,
    MAIN_TABLE_MODELS,
    METACATEGORY_NAMES,
    NORMS_NAMES,
    NORMS_LOADERS,
    OUTPUT_PATH,
    load_taxonomy_mcrae,
    load_taxonomy_mcrae_x_things,
    load_taxonomy_binder,
)
from probing_norms.extract_features_text import HFModelContextual
from probing_norms.utils import read_json, read_file


CLASSIFIER_TYPE = "linear-probe"
EMBEDDINGS_LEVEL = "concept"
SPLIT_TYPE = "repeated-k-fold"
DATASET_NAME = "things"


LOAD_TAXONOMY = {
    "McRae×THINGS": load_taxonomy_mcrae_x_things,
    "McRae++": load_taxonomy_mcrae,
    "Binder": load_taxonomy_binder,
}

st.set_page_config(layout="wide")


def load_data(model1, model2, norms_type):
    def load_score_random_features(norms_type):
        path = f"static/results/score-random-{norms_type}.json"
        return read_json(path)

    def load_result_features(model):
        path = "static/results/{}-{}-{}-{}-{}.json".format(
            CLASSIFIER_TYPE,
            EMBEDDINGS_LEVEL,
            SPLIT_TYPE,
            model,
            norms_type,
        )
        return read_json(path)

    taxonomy = LOAD_TAXONOMY[NORMS_NAMES[norms_type]]()
    scores_random = load_score_random_features(norms_type)

    results = [r for m in [model1, model2] for r in load_result_features(m)]

    for r in results:
        r["score-f1-selectivity"] = r["score-f1"] - scores_random[r["feature"]]

    df = pd.DataFrame(results)
    df["metacategory"] = df["feature"].map(taxonomy)
    df["metacategory"] = df["metacategory"].map(lambda x: METACATEGORY_NAMES.get(x, x))
    cols = ["feature", "metacategory", "model", "score-f1-selectivity"]
    df = df[cols]
    df = df.set_index(["feature", "metacategory", "model"]).unstack(-1)
    df = df.reset_index()
    df.columns = ["-".join(c for c in cols if c).strip() for cols in df.columns.values]

    return df


class Filter:
    pass


class FilterByPositive(Filter):
    def __call__(self, df):
        return df[df["is-positive"]]

    def __str__(self):
        return "positive"


class FilterByNegative(Filter):
    def __call__(self, df):
        return df[~df["is-positive"]]

    def __str__(self):
        return "negative"


class NoFilter(Filter):
    def __call__(self, df):
        return df

    def __str__(self):
        return "all"


class Sorter:
    pass


class RandomOrder(Sorter):
    def __call__(self, df):
        return df.sample(frac=1).reset_index(drop=True)

    def __str__(self):
        return "random order"


class SortByConceptName(Sorter):
    def __call__(self, df):
        return df.sort_values("concept")

    def __str__(self):
        return "concept name"


class SortByScore(Sorter):
    def __init__(self, model):
        self.model = model

    def __call__(self, df):
        return df.sort_values(self.model, ascending=False)

    def __str__(self):
        return "score {}".format(FEATURE_NAMES[self.model])


class SortByDifference(Sorter):
    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2

    def __call__(self, df):
        df["diff"] = df[self.model1] - df[self.model2]
        return df.sort_values("diff", ascending=False)

    def __str__(self):
        return "difference {} - {}".format(
            FEATURE_NAMES[self.model1],
            FEATURE_NAMES[self.model2],
        )


def show_results(
    norms_type, norms_loader, models, feature, filter, sorter, num_to_show
):
    image_names = dict(
        read_file(
            "static/things-image-names.txt",
            lambda line: line.split(),
        )
    )
    contexts = HFModelContextual.load_context("gpt4o_concept_context_sentences_v2")

    def get_path_image(concept):
        IMG_URL = "https://things-initiative.org/uploads/THINGS/images_resized/{}/{}"
        return IMG_URL.format(concept, image_names[concept])

    feature_to_concepts, _, features_selected = norms_loader()
    concepts = norms_loader.load_concepts()

    if norms_type == "binder-median":
        df_binder = load_binder_dense()
        df_binder = df_binder[df_binder["Feature"] == feature]
        df_binder = df_binder.set_index("Word")
        median = df_binder["Value"].median()
        st.write("Median rating: {:.2f}".format(median))

        def get_rating_str(concept):
            return "rating: {:.1f}".format(df_binder.loc[concept, "Value"])

    # FIXME These results were generated with get_results.py aggregate_predictions_for_demo
    # I had to use NumPy to be able to push those to GitHub and use them on Streamlit cloud.
    # Is there a better solution? Maybe host the files on GitHub releases?
    # Currently, the format is a bit too brittle.
    data = np.load("static/predictions-{}.npz".format(norms_type))
    preds = data["results"]
    models_all = data["models"].tolist()
    i_feature = features_selected.index(feature)
    i_models = (
        models_all.index(models[0]),
        models_all.index(models[1]),
    )
    preds = preds[i_feature, i_models]
    df = pd.DataFrame(preds.T, columns=models)
    df["i"] = range(len(df))
    df["concept"] = df.index.map(lambda i: concepts[i])
    df["is-positive"] = df["concept"].map(lambda c: c in feature_to_concepts[feature])

    df = filter(df)
    df = sorter(df)

    def show1(row):
        concept = row["concept"]
        is_positive_str = "✓" if row["is-positive"] else "✗"
        ss = (
            [
                "concept: {}".format(concept, is_positive_str),
                "positive: {}".format(is_positive_str),
            ]
            + ([get_rating_str(concept)] if norms_type == "binder-median" else [])
            + [
                "score {}: {:.2f}".format(FEATURE_NAMES[model], row[model])
                for model in models
            ]
        )
        s = "\n".join(["```", "\n".join(ss), "\n```"])
        st.markdown(s)
        st.image(get_path_image(concept), caption=contexts[concept][0], width=128)

    df = df.head(num_to_show)
    for group in partition_all(4, df.iterrows()):
        cols = st.columns(4)
        for col, (_, row) in zip(cols, group):
            with col:
                show1(row)


def selectboxq(col, name, options, default=None, *args, **kwargs):
    if name in st.query_params:
        default = st.query_params[name]
    if default is not None:
        index = options.index(default)
    else:
        index = 0
    selected = col.selectbox(name, options, index=index, *args, **kwargs)
    # st.query_params[name] = selected
    return selected


if __name__ == "__main__":
    norms_types = ["mcrae-x-things", "mcrae-mapped", "binder-median"]

    model_names = [FEATURE_NAMES[m] for m in MAIN_TABLE_MODELS]
    norm_types_names = [NORMS_NAMES[n] for n in norms_types]

    with st.sidebar:
        model1_name = selectboxq(st, "Model 1", model_names, "Gemma")
        model2_name = selectboxq(st, "Model 2", model_names, "DINOv2")
        norms_type_name = selectboxq(st, "Dataset", norm_types_names, "McRae×THINGS")

    if model1_name == model2_name:
        st.error("Please select two different models.")
        st.stop()

    model1 = MAIN_TABLE_MODELS[model_names.index(model1_name)]
    model2 = MAIN_TABLE_MODELS[model_names.index(model2_name)]
    norms_type = norms_types[norm_types_names.index(norms_type_name)]

    norms_loader = NORMS_LOADERS[norms_type]()
    _, _, features_selected = norms_loader()
    models = [model1, model2]

    df = load_data(model1, model2, norms_type)

    selection = alt.selection_point(fields=["metacategory"], bind="legend")

    scatter = (
        alt.Chart(df)
        .mark_circle()
        .encode(
            x=alt.X(f"score-f1-selectivity-{model1}").title(model1_name),
            y=alt.Y(f"score-f1-selectivity-{model2}").title(model2_name),
            color="metacategory",
            tooltip=["feature", "metacategory"],
            opacity=alt.when(selection).then(alt.value(1)).otherwise(alt.value(0.1)),
        )
        .properties(height=500, width=700)
        .interactive()
        .add_params(selection)
    )
    st.markdown(
        "Linear probe accuracy (F₁ selectivity score) for the two models ({} and {}) and for each of the {} attributes.".format(
            model1_name, model2_name, len(features_selected)
        )
    )
    st.altair_chart(scatter)
    st.markdown("---")

    filters = [NoFilter(), FilterByPositive(), FilterByNegative()]
    filters_str = [str(f) for f in filters]

    sorters = [
        RandomOrder(),
        SortByConceptName(),
        SortByScore(model1),
        SortByScore(model2),
        SortByDifference(model1, model2),
        SortByDifference(model2, model1),
    ]
    sorters_str = [str(s) for s in sorters]

    cols = st.columns(4)
    feature = selectboxq(cols[0], "Feature", features_selected)
    filter_by = selectboxq(cols[1], "Filter by", filters_str)
    sort_by = selectboxq(cols[2], "Sort by", sorters_str, default=sorters_str[0])
    num_to_show = cols[3].number_input(
        "Number of samples to show",
        min_value=4,
        max_value=128,
        value=12,
        step=4,
    )

    # def encode(s):
    #     return s.replace(" ", "%20")

    # query_string = "&".join(f"{}={}".format(encode(key), encode(value)) for key, value in st.query_params.items())
    # shareable_url = f"{st.get_url()}?{query_string}"
    # st.markdown(f"Shareable URL: [link]({shareable_url})")

    filter = filters[filters_str.index(filter_by)]
    sorter = sorters[sorters_str.index(sort_by)]
    st.markdown(
        "Results at sample level (per concept) for each of the two models ({} and {}). For each concept, we show an example of an image and a corresponding contextual sentence.".format(
            model1_name, model2_name
        )
    )
    show_results(norms_type, norms_loader, models, feature, filter, sorter, num_to_show)
