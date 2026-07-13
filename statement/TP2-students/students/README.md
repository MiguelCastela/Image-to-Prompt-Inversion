# TP2 Student Starter Pack

Files:

- `TP2_StarterPack_Students.ipynb`: Colab/VS Code starter notebook.
- `tp2-chosen/`: copy of the TP2 target images.
- `tp2-chosen.zip`: optional zip with the same target images.
- `outputs/`: local output folder placeholder.

Recommended Google Drive layout for Colab / VS Code Colab extension:

```text
MyDrive/GENAI_TP2/tp2-chosen/*.png
```

or:

```text
MyDrive/GENAI_TP2/tp2-chosen.zip
```

The notebook mounts Google Drive, searches these paths, extracts the zip if needed, and saves generated outputs to:

```text
MyDrive/GENAI_TP2/outputs/
```

The LCM settings match the TP2 target generation setup:

- model: `SimianLuo/LCM_Dreamshaper_v7`
- seed: parsed from target filename
- inference steps: `8`
- guidance scale: `8.0`
- `lcm_origin_steps`: `50`
- resolution: `768x768`

If Colab raises an error such as:

```text
cannot import name '_Ink' from 'PIL._typing'
```

restart the runtime/kernel and rerun the notebook from the first cell. The install cell pins `Pillow<12` to avoid that Diffusers/Pillow compatibility issue.

The install cell also pins `pandas<3` to avoid dependency conflicts with packages commonly preinstalled in Colab, such as Gradio.
