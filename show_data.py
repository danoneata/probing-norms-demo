import streamlit as st
import numpy as np

from toolz import partition_all

from probing_norms.data import DATASETS
from probing_norms.scripts.eval_norm_correlations import get_supercategories
from probing_norms.predict import NORMS_LOADERS
from probing_norms.utils import multimap


def get_image_path_online(image_file):
    _, folder, filename = image_file.split("/")
    return f"https://things-initiative.org/uploads/THINGS/images_resized/{folder}/{filename}"


st.set_page_config(layout="wide")


@st.cache_resource
def load_resources():
    dataset = DATASETS["things"]()

    norms_loader = NORMS_LOADERS["mcrae-x-things"]()
    attribute_to_concepts, _, _ = norms_loader()
    concept_and_attribute = [
        (concept, attribute)
        for attribute, concepts in attribute_to_concepts.items()
        for concept in concepts
    ]
    concept_to_attributes = multimap(concept_and_attribute)

    supercategory_to_concepts = get_supercategories()
    concept_and_supercategory = [
        (concept, supercategory)
        for supercategory, concepts in supercategory_to_concepts.items()
        for concept in concepts
    ]
    concept_to_supercategories = multimap(concept_and_supercategory)

    return dataset, concept_to_attributes, concept_to_supercategories


dataset, concept_to_attributes, concept_to_supercategories = load_resources()
concepts = sorted(dataset.class_to_label.keys())


def get_supercategories_str(concept):
    return ", ".join(concept_to_supercategories.get(concept, []))

cols = st.columns([1, 1, 4])

with cols[0]:
    st.markdown("### Concept")
    concept = cols[0].selectbox("Select a concept", concepts)
    label = dataset.class_to_label[concept]
    st.markdown("### Supercategories")
    st.markdown(get_supercategories_str(concept))

with cols[1]:
    st.markdown("### Attributes")
    st.markdown("\n".join(f"- {a}" for a in concept_to_attributes.get(concept, [])))

with cols[2]:
    st.markdown("### Images")
    idxs, = np.where(np.array(dataset.labels) == label)

    num_cols = 6
    for group in partition_all(num_cols, idxs):
        colss = st.columns(num_cols)
        for i, idx in enumerate(group):
            image_file = dataset.image_files[idx]
            colss[i].image(get_image_path_online(image_file))
