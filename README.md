# README

Our recipe is adapted from the original DASB recipe.  
First, download DASB from [https://github.com/speechbrain/benchmarks](https://github.com/speechbrain/benchmarks) and follow the installation instructions to set up the required packages.  
Then download the necessary datasets from the following links:  

- LibriSpeech: [https://www.openslr.org/12](https://www.openslr.org/12)  
- LibriLight: [https://github.com/facebookresearch/libri-light/blob/main/data_preparation/README.md](https://github.com/facebookresearch/libri-light/blob/main/data_preparation/README.md)  
- Common Voice: [https://commonvoice.mozilla.org](https://commonvoice.mozilla.org)  

After setup, move the `DASBASREval` folder into:  
```
/your_DASB_root/benchmarks/benchmarks
```  

Inside `DASB_Test`, you will find two subfolders, `LibriSpeech` and `CommonVoice`.  
You can modify the YAML files under `ASR/LSTM/hparams` to run inference for **CATT-RPG** and **FT-EMB**.  

---

### DSU files  
In the YAML configuration, you will see entries such as:  
```yaml
ssl_dsu: /local/LibriSpeech100_hubert_dsus.pt
# or
ssl_dsu: /local/CommonVoice_eu_hubert_dsus.pt
```  
These point to pre-extracted DSU files (saved with dictionary keys as filenames) to speed up inference.  
Since these files are large, we provide them here:  
[Google Drive link](https://drive.google.com/drive/folders/1Wey4Y8Man0SqyVXIZmhqewAEQtW3o8PH?usp=sharing)  

For Common Voice, you can switch the language with:  
```yaml
language: eu
```  

---

### Inference example  
We provide an example using **HuBERT trained on LS100** as the extractor model for inference on LibriSpeech:  
```bash
CUDA_VISIBLE_DEVICES=2 python LibriSpeech/ASR/LSTM/eval_cattrpg.py     LibriSpeech/ASR/LSTM/hparams/eval_catt_rpg.yaml     --output_folder outputs/LibriSpeech/hubert_cattrpg_ls100     --data_folder /mnt/md0/dataset/LibriSpeech
```  

- `output_folder`: Path to our pretrained models. Due to size limits, we release them here:  
  [Google Drive link](https://drive.google.com/drive/folders/1Wey4Y8Man0SqyVXIZmhqewAEQtW3o8PH?usp=sharing)  

- `data_folder`: Directory of audio files (LibriSpeech / LibriLight / Common Voice).  

---

### FT-EMB notes  
To run FT-EMB, use `eval_ft_emb.yaml`.  
When using **WavLM as the extractor**, you need to manually update:  
```
DASBASREval/speechbrain/speechbrain/lobes/models/huggingface_transformers/discrete_ssl.py
```  
Set the `vocoder_repo_id` parameter to:  
```python
"speechbrain/hifigan-wavlm-k1000-LibriTTS"
```  

---

### Released inference recipes  
We release inference recipes under our paper’s standard setup for:  

- LibriSpeech-100 (LS100)  
- LibriSpeech-960 (LS960)  
- LibriLight-10h (LL10)  
- Common Voice (Basque, Welsh, Swedish)  

Both **CATT-RPG** and **FT-EMB** recipes are provided.  
