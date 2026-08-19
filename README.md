# MAGIC-HRI: Multimodal Activity, Gesture, and Intention Collection for HRI

## Overview
The **MAGIC-HRI (Multimodal Activity, Gesture, and Intention Collection for HRI)** dataset was developed at the Applied Robotics Laboratory at UTFPR. It contains Electromyography (EMG) and Inertial Measurement Unit (IMU) signals recorded during human-robot collaboration tasks. It is designed to support research in safe Human-Robot Interaction (HRI), specifically targeting human action estimation and prediction in industrial settings.

## Experimental Setup
The data acquisition setup consists of a wearable gesture-recognition interface and a collaborative robotic arm:
* **Wearable Interface (Myo Armband - Model MYOD5):** A non-invasive device from Thalmic Labs that collects data via eight EMG electrodes. It features IMU sensors, including a 3D accelerometer and a 3D gyroscope. 
* *Sampling Rates:* EMG samples up to 200Hz; IMU samples up to 50Hz.
* *Dataset Output:* All signals are recorded and synchronized at a frequency of **50Hz**.
* *Placement:* Right forearm, with the channel 3 electrode on the flexor carpi ulnaris muscle. All participants were right-handed.
* **Collaborative Robot (OMRON TM5-700):** A 6-axis collaborative robotic system from Techman Robot, utilized to deliver objects to the subjects and receive objects from them during data acquisition.

## Dataset Description
The dataset includes recordings from **11 participants** performing a series of predefined tasks and gestures. A total of **53 movement classes** were defined. Each participant contributed 10 samples per class, resulting in 530 samples and approximately 57 minutes of raw data per individual.

### Movement Categories
The movement classes are categorized into five main groups:
1.  **Numbers:** Cardinal numbers 0 through 9 represented in Brazilian Sign Language (LIBRAS).
2.  **Gestures:** Basic hand configurations (spread fingers, open palm, wave in and out, thumb-to-middle, closed hand, and thumbs up).
3.  **Object Interaction:** Holding, giving objects or tools to a robot, and picking up objects or tools from a robot.
4.  **Tool Manipulation:** Screwing and unscrewing tasks (using an electric screwdriver, an adjustable wrench, or a Phillips screwdriver) and active use of tools (including a hot glue gun and a heat gun).
5.  **Generic Movements:** This category includes idle states, empty-hand movements, and general assembly tasks. Those movements were recorded with the possibility of avoiding noise in the classification process in mind.

---

## Repository Structure
The repository is organized into two main directories to neatly separate the processing scripts from the raw data hierarchy:

* **`Codes/`**: This directory contains the Python scripts (`feature_extraction.py` and `grid_search.py`) provided as baseline examples for utilizing the dataset.
* **`Dataset/`**: This directory contains the raw data files, structured hierarchically for easy programmatic parsing (e.g., using `os.walk`):
    * **`Aquisition_01/` to `Aquisition_11/`**: 11 root folders representing each participant's individual acquisition session.
        * **`aw/`, `bl/`, `bx/`, `es/`, `gg/`, `gs/`, `ha/`, `hg/`, `nu/`, `pl/`, `ps/`, `sc/`**: Inside each acquisition folder, the data is further grouped into 12 subdirectories corresponding to the tool or gesture category used in that trial.
            * **`.csv` files**: Inside these subdirectories are the individual comma-separated values files containing the synchronized EMG and IMU data.

---

## File Nomenclature
Data files are named using the following convention to easily identify the contents of each trial:

**`TT_MM_SS_PP_NN`**

* **`TT`**: Tool name or gesture category
* **`MM`**: Movement name
* **`SS`**: Period of acquisition (e.g., 01, 02, ...)
* **`PP`**: Person/Participant number (e.g., 01, 02, ...)
* **`NN`**: Number of the movement sample (e.g., 01, 02, ...)

### Tool / Gesture Codes (`TT`)
| Code | Description | Code | Description |
| :--- | :--- | :--- | :--- |
| **`es`** | Electric screwdriver | **`bx`** | Box |
| **`gg`** | Hot glue gun | **`sc`** | Screw |
| **`aw`** | Adjustable wrench | **`hg`** | Heat gun |
| **`ps`** | Phillips screwdriver | **`ha`** | Hand |
| **`bl`** | Block | **`gs`** | Gesture |
| **`pl`** | Planke | **`nu`** | Numbers |

### Movement Codes (`MM`)
| Code | Description | Code | Description |
| :--- | :--- | :--- | :--- |
| **`hd`** | Hold | **`op`** | Open palm |
| **`gv`** | Give | **`wi`** | Wave in |
| **`pu`** | Pick up | **`wo`** | Wave out |
| **`sw`** | Screwing | **`tm`** | Thumb to middle |
| **`un`** | Unscrewing | **`hc`** | Hand closed |
| **`em`** | Empty | **`tu`** | Thumbs up |
| **`id`** | Idle | **`us`** | Using |
| **`sf`** | Spread fingers | **`as`** | Assembling |

---

## Task Durations
The duration of each movement class varies depending on the complexity of the task (ranging from 3s to 30s):

| Task | Sample Period (s) | Task | Sample Period (s) |
| :--- | :--- | :--- | :--- |
| All Hold | 3 | All Numbers | 3 |
| All Give | 4 | Screw Electric Screwdriver | 10 |
| All Pick up | 4 | Unscrew Electric Screwdriver| 10 |
| Screw Adjustable Wrench | 30 | Screw Phillips Screwdriver | 15 |
| Unscrew Adjustable Wrench | 30 | Unscrew Phillips Screwdriver| 15 |
| Hot Glue Gun Using | 25 | Hands empty | 10 |
| Heat Gun Using | 15 | Idle | 10 |
| All Gestures | 3 | Generic Assemble | 30 |

---

## Participant Demographics
Data was collected from 11 healthy, right-handed individuals (9 males, 2 females) with a mean age of 33 years (range: 22 to 54).

| ID | Age | Sex | ID | Age | Sex |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **01** | 29 | M | **07** | 22 | M |
| **02** | 40 | M | **08** | 26 | M |
| **03** | 39 | M | **09** | 54 | M |
| **04** | 32 | M | **10** | 25 | F |
| **05** | 39 | M | **11** | 37 | F |
| **06** | 24 | M | | | |

---

## Code Examples
The `Codes` directory contains python scripts intended to serve as a baseline inspiration for utilizing the MAGIC-HRI dataset in machine learning applications. They demonstrate a pipeline from raw data handling to model evaluation.

### 1. `feature_extraction.py`
This script is responsible for traversing the dataset directory and preparing the raw signals. Its main steps are:
* **Muscle Activation Detection:** Automatically finds the active segment of the movement by calculating the average energy across the EMG channels.
* **Sliding Window:** Applies a sliding window approach (e.g., 1-second windows with 90% overlap) over the active signal segment.
* **Feature Engineering:** Extracts a rich set of mathematical, statistical, and spectral features (such as RMS, zero-crossings, sample entropy, median frequency, etc.) from both the EMG and IMU data.
* **Data Output:** Saves all the extracted features into a structured `.csv` file, making it ready to be fed into classification algorithms.

### 2. `grid_search.py`
This script takes the extracted features and runs a comprehensive machine learning evaluation. Its main steps are:
* **Data Preparation:** Loads the feature `.csv` file, filters necessary columns, cleans the data, and applies scaling (`StandardScaler`).
* **Hyperparameter Tuning:** Performs a robust Grid Search with Cross-Validation across 10 different classic classifiers (including Random Forest, SVM, KNN, AdaBoost, and MLP Neural Networks) to find the most accurate configurations.
* **Automated Reporting:** Incrementally updates a `.csv` file with the results of all combinations tested and generates a clean `.txt` report showing the top 3 best hyperparameter combinations for each trained model.