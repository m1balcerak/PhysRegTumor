# PhysRegTumor
**Physics-Regularized Multi-Modal Image Assimilation for Brain Tumor Localization**  
(NeurIPS 2024)  
**UNDER CONSTRUCTION v.01**

---

## Required Packages

To run this project, make sure you have the necessary dependencies installed. You can use the provided `requirements_PhysRegTumor.txt` file to install all required packages.

To install the dependencies, run the following command:

```bash
pip install -r requirements_PhysRegTumor.txt
```

Alternatively, if you prefer using `conda`:

```bash
conda create --name physregtumor --file requirements_PhysRegTumor.txt
conda activate physregtumor
```

---

## Running the Code

To run the project, follow these steps:

1. Ensure that your environment is set up with the required dependencies (as outlined above).
2. Open your terminal and navigate to the project directory. Make sure you have execute permission for the script. If needed, update the permissions with the following command:

   ```bash
   chmod +x run_instance.sh
   ```

3. Inspect the `run_instance.sh` script to make sure that you’ve selected the correct paths to your dataset and the desired patient code(s) for calculation. The relevant lines in the script will look like this:

   ```bash
   WM_FILE_PATH="/path_to_data/data_${code}/t1_wm.nii.gz"
   GM_FILE_PATH="/path_to_data/data_${code}/t1_gm.nii.gz"
   CSF_FILE_PATH="/path_to_data/data_${code}/t1_csf.nii.gz"
   SEGM_FILE_PATH="/path_to_data/data_${code}/segm.nii.gz"
   PET_FILE_PATH="/path_to_data/data_${code}/FET.nii.gz"
   ```

   - Replace `/path_to_data` with the actual path where your dataset is stored.
   - Set `${code}` to the code of the patient you want to process.

4. The output of the script will be saved in a directory named `FK_${code}`, containing all results for the selected patient.

5. If you don't have PET imaging available, you can still run the framework, but make sure to set the PET weight to zero in the `PhysRegTumor.py` script by adjusting the following line:

   ```python
   pet_w = 0
   ```

6. Once you've made the necessary adjustments, run the bash script with the following command:

   ```bash
   bash run_instance.sh
   ```

---

## Data

This framework requires a dataset to work with. The dataset used in this project can be found at the following link:

[GliODIL Dataset on Hugging Face](https://huggingface.co/datasets/m1balcerak/GliODIL)

Ensure you download and properly configure the dataset before running the script.

---
