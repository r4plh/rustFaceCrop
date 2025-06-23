### Instructions.md file 

# Important information regarding dataset used.
* A computer running a modern Linux distribution (e.g., Ubuntu, Fedora).
* The **`WIDER FACE` Dataset**. This project is configured to use the validation set (This is how I used and generated cropped human face images.)
    1.  Go to the dataset repository on Hugging Face: [CUHK-CSE/wider\_face](https://huggingface.co/datasets/CUHK-CSE/wider_face).
    2.  Download the `WIDER_val.zip` file and extract it.
    3.  You will get a folder named `WIDER_val`. This is the folder you need.

*(Note: I used this dataset to generate over 18,000 face crops from its ~3,200 validation images.)*

### Package Contents 

Package to be downloaded - [Download the package for executing binary file for this program](https://github.com/r4plh/rustFaceCrop/releases/latest)

* `rust-yolo-linux`: The executable binary file.
* `yolov11n-face.onnx`: The required ONNX neural network model.

### Instructions

1.  Create a directory and place the `rust-yolo-linux` binary, the `yolov11n-face.onnx` model, and the `WIDER_val` folder inside it. Your folder structure must look like this:

    ```
    /my_project_folder/
    ├── rust-yolo-linux
    ├── yolov11n-face.onnx
    └── WIDER_val or any sample folder containing relavent images/
        ├── 0--Parade/
        │   ├── 0_Parade_marchingband_1_1.jpg
        │   └── ...
        └── 1--Handshaking/
            └── ...
    ```

2.  Open a terminal in that directory (`my_project_folder`).

3.  Make the binary executable. This is a one-time command to grant it permission to run.
    ```bash
    chmod +x ./rust-yolo-linux
    ```

4.  Execute the program.
    ```bash
    ./rust-yolo-linux
    ```

The program will begin processing all images within `WIDER_val`. It will create a new directory named `wider_face_crops` and save the output there. Progress will be printed to the terminal.

### (Optional) For Developers: Testing from Source Code (To Test)

If you want to compile the code yourself or test it with a custom folder of images, follow these steps.

1.  **Setup:** Make sure you have the Rust toolchain installed.
2.  **Modify Source:**
    * Open the `src/main.rs` file.
    * In the `main` function, locate the following line:
        ```rust
        let images = load_local_images("WIDER_val")?;
        ```
    * Change the folder name `"WIDER_val"` to your custom folder's name. For example, if you have a folder named `sample_images`, change the line to, I have added sample_images folder to test for few images which are there in sample images because WIDER_FACE_VAL is a big dataset:
        ```rust
        let images = load_local_images("sample_images")?;
        ```
3.  **Run the Code:** Execute the project using Cargo.
    ```bash
    cargo run
    ```
    Cargo will compile and run the program, creating the `wider_face_crops` directory with the results from your custom folder.
