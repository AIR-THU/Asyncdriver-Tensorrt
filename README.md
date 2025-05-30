<div align="center">   
  
# Asynchronous Large Language Model Enhanced Planner for Autonomous Driving
</div>

<h3 align="center">
  <a href="https://arxiv.org/abs/2406.14556">arXiv</a> |
  <a href="https://huggingface.co/datasets/Member22335/AsyncDriver">Dataset</a>
</h3>

Official implementation of the **ECCV 2024** paper **Asynchronous Large Language Model Enhanced Planner for Autonomous Driving**.

## Getting Started

### 1. Installation

#### Step 1: Download NuPlan Dataset

- Follow the [official instructions](https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html) to download the NuPlan dataset.

#### Step 2: Set Environment Variables

Make sure to set the following environment variables correctly to point to the NuPlan datase:

```
NUPLAN_MAPS_ROOT=path/to/nuplan/dataset/maps
NUPLAN_DATA_ROOT=path/to/nuplan/dataset
NUPLAN_EXP_ROOT=path/to/nuplan/exp
```

#### Step 3: Clone the Repository

Clone this repository and navigate to the project directory:

```
git clone https://github.com/memberRE/AsyncDriver.git && cd AsyncDriver
```
If you want to deploy on NVIDIA Jetson Orin, please switch to the orin_tensorrt branch:

```
git checkout orin_tensorrt
```

#### Step 4: Set up the Conda Environment

- **For NVIDIA Jetson Orin (ARM64):**

  First, install the JetPack SDK:

  ```bash
  sudo apt-get update && sudo apt install nvidia-jetpack
  ```

  Then check your device info:

  ```bash
  jetson_release
  ```

  Example output:
  ```
  Model: Jetson AGX Orin Developer Kit - Jetpack 5.1.2 [L4T 35.4.1]
  Power Mode: MODE_50W
  CUDA: 11.4.315
  cuDNN: 8.6.0.166
  TensorRT: 8.5.2.2
  OpenCV: 4.5.4 - with CUDA: NO
  ```

- **Create the Jetson Conda Environment:**

  Manually create a Conda environment for Python 3.8 (ARM compatible):

  ```bash
  conda create -n jetson38 python=3.8 -y
  conda activate jetson38
  ```

- **Install Additional Dependencies:**

  After activating the environment, run the ARM-specific setup script:

  ```bash
  bash env_arm.sh
  ```
- **OpenCV Loading Issue (LD_PRELOAD Fix)**

  If you encounter OpenCV runtime loading errors, set the following environment variable before running your program:

  ```bash
  export LD_PRELOAD="/lib/aarch64-linux-gnu/libffi.so.7.1.0 \
  /lib/aarch64-linux-gnu/libgio-2.0.so.0"
  ```
---

- **For x86_64:**

  Create a Conda environment based on the provided `environment.yml` file:

  ```bash
  conda env create -f environment.yml
  ```

- **Install Additional Dependencies:**

  After setting up the Conda environment, install the additional dependencies listed in the `requirements_asyncdriver.txt`:

  ```bash
  pip install -r requirements_asyncdriver.txt
  ```

  > *Note:* If you encounter any issues with dependencies, refer to the `environment_all.yaml` for a complete list of packages.

#### Step 5: Download Checkpoints and AsyncDriver Dataset 

- Download the [**PDM checkpoint**](https://github.com/autonomousvision/tuplan_garage), and update the necessary file paths in the configuration (although this checkpoint is not actively used in the current version).
- Download the [**llama-2-13b-chat-hf**](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf).
- Download the [training data](https://huggingface.co/datasets/Member22335/AsyncDriver/resolve/main/stage1_train_180k_processed.json) and [validate data](https://huggingface.co/datasets/Member22335/AsyncDriver/resolve/main/stage1_val_20k_processed.json) and update the `map_info` field in the JSON files to the corresponding file's absolute path.

### 2. Evaluation

To evaluate the model, use the following command:

~~~
bash train_script/inference/asyncdriver_infer.sh <gpuid> <scenario_type_id>
~~~

> `<scenario_type_id>` is a value between [0-13], representing 14 different scenario types. Replace all `path/to` placeholders in the scripts with actual paths.

To evaluate the model with asynchronous inference, use:

~~~
bash train_script/inference/with_interval.sh <gpuid> <scenario_type_id> <interval>
~~~

> `<interval>` defines the inference interval between LLM and Real-time Planner, and it should be set to a value between [0, 149].

To evaluate the model with `pdm_scorer`, use:

~~~
bash train_script/inference/with_pdm_scorer.sh <gpuid> <scenario_type_id>
~~~


> *Note:* Update `nuplan/planning/script/config/simulation/planner/llama4drive_lora_ins_wo_stop_refine.yaml` at line 58 with the correct PDM checkpoint path. This path is required for instantiation but is not used during execution.

If you encounter issues with the planner not being found, modify the following line:

- Change `train_script/inference/simulator_llama4drive.py` from line 83 to line 84.

Training checkpoints is available for [download](https://drive.google.com/file/d/17TLnwgp7T6ke67kgSqnc2dhTCZn83W6a/view?usp=drive_link).


### 3. Evaluation with TensorRT and ONNX

#### Step 1: Export the ONNX File

Run the following command to export the merged LoRA model to ONNX:

```bash
python export_onnx.py \
  --model_path /path/to/base_model \
  --lora_path /path/to/lora_model \
  --onnx_path /path/to/output_model.onnx
```

#### Step 2: Convert ONNX to TensorRT

Navigate to the converter directory, compile the binary, and generate the TensorRT engine:

```bash
cd onnx_to_tensorrt
mkdir build && cd build
cmake ..
make
./onnx_to_tensorrt /path/to/model.onnx /path/to/model.engine
```

#### Step 3: Modify Inference Script

Edit the script `train_script/inference/asyncdriver_infer.sh` and configure the following variables:

- `onnx_model_path`: path to the exported ONNX model.
- `tensorrt_model_path`: path to the generated TensorRT engine.
- `inference_model_type`: one of `torch`, `onnx`, or `tensorrt`.

#### Step 4: Run Evaluation

Follow the steps in [Section 2: Evaluation](#2-evaluation) to run model inference using the configured backend.

---

> **Note for NVIDIA Jetson Orin (ARM64):**
>
> Due to limited support for LoRA fine-tuning in JetPack 5.1.2, it is recommended to **export the ONNX model on an x86 host machine** and then transfer the exported model to the NVIDIA Jetson Orin device.
>
> Once transferred, you can then generate the TensorRT engine by the steps above.

#### LLaMA Inference Performance on NVIDIA Jetson AGX Orin

The following table presents the inference latency of the LoRA-finetuned LLaMA component used in **AsyncDriver**, tested on NVIDIA **Jetson AGX Orin** under **mode_50W** power mode. The comparison includes different inference backends and precision settings.

| Inference Backend           | Time (s) |
|-----------------------------|---------:|
| PyTorch (Linear4bit)        |   0.3250 |
| ONNX Runtime (FP16)         |   0.1265 |
| ONNX Runtime (FP32)         |   0.1960 |
| TensorRT (FP16)             |   0.1016 |
| TensorRT (FP32)             |   0.2149 |

### 4. Training

The training process involves multiple stages:

- **Train GameFormer:**

~~~
python train_script/train_gameformer.py --train_set path/to/stage1_train_180k_processed.json --valid_set stage1_val_20k_processed.json
~~~

- **Train Planning-QA:**

~~~
bash train_script/train_qa/train_driveqa.sh <gpu_ids>
~~~

- **Train Reasoning1K:**

~~~
bash train_script/train_qa/train_mixed_desion_qa.sh <gpu_ids>
~~~

- **Final stage:**

~~~
bash train_script/train_from_scratch/llm_load_pretrain_lora_gameformer.sh <gpu_ids>
~~~

> *Note:* Make sure to replace all `path/to` placeholders in the scripts with actual paths.

## Citation
If you find this repository useful for your research, please consider giving us a star 🌟 and citing our paper.

~~~
@inproceedings{chen2024asynchronous,
 author = {Yuan Chen, Zi-han Ding, Ziqin Wang, Yan Wang, Lijun Zhang, Si Liu},
 title = {Asynchronous Large Language Model Enhanced Planner for Autonomous Driving},
 booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
 year = {2024}}
~~~

## Acknowledgements

Some of the codes are built upon [nuplan-devkit](https://github.com/motional/nuplan-devkit), [GameFormer](https://github.com/MCZhi/GameFormer-Planner), [tuplan_garage](https://github.com/autonomousvision/tuplan_garage) and [llama](https://github.com/meta-llama/llama). Thanks them for their great works!


