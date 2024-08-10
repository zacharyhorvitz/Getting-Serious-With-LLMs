<h1 align='center'> :satisfied: &rarr; Getting Serious about Humor with LLMs &rarr; :neutral_face: </h1>

Code for **[Getting Serious about Humor: Crafting Humor Datasets with Unfunny Large Language Models (ACL 2024)](https://arxiv.org/abs/2403.00794)**.

## Structure

- [**`data_generation`**](./data_generation)  
  Contains logic for generating synthetic data and the corresponding outputs.

- [**`datasets`**](./datasets)  
  Includes original human data for both the Unfun and English-Hindi tweets humor datasets.

- [**`humor_detection`**](./humor_detection)  
  Houses the logic for training humor classifiers using both synthetic and human humor data.

- [**`evaluation`**](./evaluation)  
  Contains human evaluation results, automatic evaluation metrics, and relevant scripts.

- [**`utils`**](./utils)  
  Utilities for interacting with LLM APIs and handling various file I/O operations.

## Getting Started

1. **Set up a Python environment:**  
   Create and activate a Python environment using `conda` or `venv`. Python 3.9 is recommended.
   
2. **Install dependencies:**  
   After activating your environment, install the required packages by running:
   
   ```bash
   pip install -r requirements.txt
   ```

## TODO

- [ ] Trim unnecessary packages from `requirements.txt`.
- [ ] Include/Improve readability of:
  - [ ] Data preprocessing scripts
  - [ ] English-Hindi data generation scripts

## Citation

```bibtex
@misc{horvitz2024gettinghumorcraftinghumor,
      title={Getting Serious about Humor: Crafting Humor Datasets with Unfunny Large Language Models}, 
      author={Zachary Horvitz and Jingru Chen and Rahul Aditya and Harshvardhan Srivastava and Robert West and Zhou Yu and Kathleen McKeown},
      year={2024},
      eprint={2403.00794},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2403.00794}, 
}
```
