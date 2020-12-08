# A Convolutional Network for Face Mask Detection - Deliverable 2
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18l23Rah-Qfpubf5FXBOh2rB42gFfO8iu?usp=sharing)

## Files

- `Dataset.zip`
- `face_mask_detector_cnn.ipynb`

## Requirements

- [Google Colab](https://colab.research.google.com/)
- [Google Drive](https://drive.google.com/)

## Instructions

- Create an empty directory `COMP-6721` in the root of your google drive
- Go to colab and upload the file `face_mask_detector_cnn.ipynb` to open the jupyter notebook or go to this [Project Link](https://colab.research.google.com/drive/1pwp6V_l2gt0dQCdvLKBCiCVieCOzazkV?usp=sharing) to open file in colab automatically
- Run the first cell to mount your google drive folder, it would prompt you with a URL to give access permissions to your google drive
- Copy the authorisation code from that URL in cell's output to finish mounting the drive.
- The project uses your account's google drive to download the dataset and uses `COMP-6721` as its working directory.
- Run each cell of the notebook on colab one after the other
- There is an option to run all cells at once in google colab, to do that go to `Runtime -> Run all (Ctrl +F9)`
- Each cell is annotated with description of what the cell does
- For subsequent runs don't run the cell `Loading the dataset` as all the images would have been downloaded in your first run.

### Alt. Method:
- Use Anaconda Navigator, use the root environment.
- Click _Open with Jupyter Notebook_
- Navigate to `face_mask_detector_cnn.ipynb` and run the cells as mentioned below.

## Training the model

> Note: Make sure to run the previous cells before running this

- Run the cell with description as `Run the k-fold training and evaluation`.

## Testing the model

> Note: Make sure to run the previous cells before running this

- Run the cell with description as `Predict the category of an image taken from a test dataset`.
