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

supercategory_to_concepts = get_supercategories()
concept_and_supercategory = [
    (concept, supercategory)
    for supercategory, concepts in supercategory_to_concepts.items()
    for concept in concepts
]
concept_to_supercategories = multimap(concept_and_supercategory)

def get_supercategories_str(concept):
    return ", ".join(concept_to_supercategories.get(concept, []))

dataset = DATASETS["things"]()
norms_loader = NORMS_LOADERS["mcrae-x-things"]()
attribute_to_concepts, _, attributes = norms_loader()
concept_and_attribute = [
    (concept, attribute)
    for attribute, concepts in attribute_to_concepts.items()
    for concept in concepts
]
concept_to_attributes = multimap(concept_and_attribute)

concepts = dataset.class_to_label.keys()
concepts = sorted(concepts)

cols = st.columns([1, 1, 4])

cols[0].markdown("### Concept")
concept = cols[0].selectbox("Select a concept", concepts)
label = dataset.class_to_label[concept]
cols[0].markdown("### Supercategories")
cols[0].markdown(get_supercategories_str(concept))

cols[1].markdown("### Attributes")
cols[1].markdown("\n".join(f"- {a}" for a in concept_to_attributes.get(concept, [])))

cols[2].markdown("### Images")
idxs, = np.where(np.array(dataset.labels) == label)

num_cols = 6
for group in partition_all(num_cols, idxs):
    colss = cols[2].columns(num_cols)
    for i, idx in enumerate(group):
        image_file = dataset.image_files[idx]
        colss[i].image(get_image_path_online(image_file))
