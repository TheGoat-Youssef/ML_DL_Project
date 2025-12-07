import streamlit as st
from pathlib import Path
from src.processing import config
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.title(" Répartition des classes (RAW)")

    counts = {cls: len(list((config.RAW_DIR/"train"/cls).glob("*"))) for cls in config.CLASS_NAMES}
    
    fig, ax = plt.subplots(figsize=(10,4))
    sns.barplot(x=list(counts.keys()), y=list(counts.values()), ax=ax)
    plt.title("Distribution des classes")
    st.pyplot(fig)
