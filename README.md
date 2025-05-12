# Image-segmentation-for-medical-images
CT Scan Image processing and Lung segmentation

Project Description

This project provides a Python-based toolkit for processing and analyzing medical Computed Tomography (CT) scans, with a particular focus on lung imagery. The codebase includes functionalities for loading DICOM images, converting them to Hounsfield Units (HU), resampling images, generating HU distribution histograms, displaying individual scan slices, segmenting the lung region in 3D scans, and normalizing pixel values. This code is valuable for researchers and developers in the medical image analysis field, offering a foundation that can be extended for more complex applications such as detecting and diagnosing pulmonary diseases.

Key Features

This code offers a set of fundamental functions for CT image processing, with key features including:

*   *DICOM Image Loading:* Ability to read a series of DICOM files from a specified folder, sorting them based on image position and correctly ordering the slices. Slice thickness is also automatically calculated.
*   *Pixel to Hounsfield Unit (HU) Conversion:* Transforms raw pixel values from DICOM images into Hounsfield Units, a standard scale for tissue density in CT imaging. RescaleIntercept and RescaleSlope values are handled to ensure accurate conversion.
*   *Image Resampling:* Modifies the voxel spacing of 3D images to new, specified values. This ensures uniform voxel size across different scans, facilitating analysis and comparison.
*   *Data Visualization:* Includes functions to display the distribution of Hounsfield Units using a histogram and to show 2D slices from the CT scan.
*   *Lung Segmentation:* Implements an algorithm to identify and extract the lung region from 3D CT scans. This process involves initial thresholding, connected component labeling, removal of undesired structures, and an option to fill internal lung structures.
*   *3D Rendering:* Creates a 3D surface representation of the segmented lung or any other part of the image using the Marching Cubes algorithm, allowing for better visualization of anatomical structures.
*   *Image Normalization:* Scales Hounsfield Unit values to a specific range (typically between 0 and 1), a common preprocessing step in machine learning and image processing applications.
*   *Zero-centering:* Adjusts pixel values so that their mean is close to zero, another data preprocessing step.

 Requirements and Dependencies

To run this code successfully, you will need to install the following libraries in your Python environment:

*   *NumPy:* Essential library for scientific computing in Python, used for handling arrays and numerical operations.
*   *Pandas:* Powerful library for data processing and analysis, primarily used here for reading CSV files (though the current code doesn't directly use it for DICOM processing).
*   *Pydicom:* Specialized library for reading and handling DICOM files.
*   *Scipy:* Library containing various modules for scientific and engineering algorithms, used here for the scipy.ndimage.interpolation.zoom function for image resampling.
*   *Matplotlib:* Comprehensive library for creating static, animated, and interactive visualizations in Python. Used for displaying histograms, image slices, and 3D plots.
*   *Scikit-image:* A collection of algorithms for image processing, used here for skimage.measure.marching_cubes, skimage.measure.label, and skimage.morphology functions.

You can install these libraries using the pip package manager:

bash
pip install numpy pandas pydicom scipy matplotlib scikit-image


Installation Steps

1.  *Clone the Repository (if available on GitHub):*
    bash
    git clone <repository_url>
    cd <repository_directory>
    
    Alternatively, download the code files directly.
2.  *Install Dependencies:* Ensure all the libraries listed above are installed in your environment. You can use the command pip install -r requirements.txt if a requirements.txt file is provided.
3.  *Set Up Input Data Folder:* The code defines a variable INPUT_FOLDER that points to the path of a folder containing DICOM images of patients. Make sure to modify this path to point to the correct location of your data. This folder should contain subfolders, where each subfolder represents a single patient and contains their series of DICOM files.
    python
    INPUT_FOLDER = '../input/sample_images/' # Modify this path
    

How to Use the Code

The provided code demonstrates a typical workflow for processing CT scan images for a single patient. You can adapt this code to process multiple patients or integrate it into larger applications.

1.  *Load Patient Data:*
    The code starts by defining the input folder and reading the patient names (subfolders). It then loads the DICOM images for the first patient using the load_scan function.
    python
    patients = os.listdir(INPUT_FOLDER)
    patients.sort()
    first_patient_scan = load_scan(INPUT_FOLDER + patients[0])
    
2.  *Convert Pixels to Hounsfield Units:*
    The raw pixel data is converted to Hounsfield Units using the get_pixels_hu function.
    python
    first_patient_pixels_hu = get_pixels_hu(first_patient_scan)
    
3.  *Display HU Distribution and Image Slice:*
    You can display the HU histogram and a specific image slice using matplotlib.
    python
    plt.hist(first_patient_pixels_hu.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()

    plt.imshow(first_patient_pixels_hu[80], cmap=plt.cm.gray)
    plt.show()
    
4.  *Resample Image:*
    The image is resampled to a new voxel spacing (e.g., 1x1x1 mm) using the resample function.
    python
    pix_resampled, spacing = resample(first_patient_pixels_hu, first_patient_scan, [1,1,1])
    print(f"Shape before resampling\t{first_patient_pixels_hu.shape}")
    print(f"Shape after resampling\t{pix_resampled.shape}")
    
5.  *3D Plotting:*
    A 3D representation of the resampled image (or a part of it) can be displayed using the plot_3d function. You can specify a threshold to display specific structures (e.g., bones at a threshold of 400 HU).
    python
    plot_3d(pix_resampled, 400) # Display high-density structures like bones
    
6.  *Lung Segmentation:*
    The lungs are segmented using the segment_lung_mask function. You can control whether to fill internal lung structures.
    python
    segmented_lungs = segment_lung_mask(pix_resampled, fill_lung_structures=False)
    segmented_lungs_fill = segment_lung_mask(pix_resampled, fill_lung_structures=True)
    
7.  *Display Segmented Lung in 3D:*
    The segmented lung (with and without filling) can be displayed using the plot_3d function.
    python
    plot_3d(segmented_lungs, 0)
    plot_3d(segmented_lungs_fill, 0)
    
8.  *Normalization and Zero-Centering (Optional):*
    Pixel values can be normalized and zero-centered using the normalize and zero_center functions as additional preprocessing steps.
    python
    normalized_image = normalize(pix_resampled)
    zero_centered_image = zero_center(normalized_image)
    

code Structure and Key Functions

The code consists of a set of functions that perform specific tasks in the CT image processing pipeline:

*   load_scan(path): Loads all DICOM slices from the specified path, sorts them, and calculates slice thickness.
*   get_pixels_hu(slices): Converts the pixel array from DICOM images to Hounsfield Units.
*   resample(image, scan, new_spacing=[1,1,1]): Resamples the 3D image to a new voxel spacing.
*   plot_3d(image, threshold=-300): Creates a 3D surface plot of the image at a specified threshold using Marching Cubes.
*   largest_label_volume(im, bg=-1): A helper function used in lung segmentation to identify the largest connected component (label) in a labeled image, ignoring the background.
*   segment_lung_mask(image, fill_lung_structures=True): Segments the lung region in the 3D image. Steps include: binarizing the image based on a threshold, identifying air components, filling air around the patient, then (optionally) filling internal lung structures, and finally removing other small air pockets within the body.
*   normalize(image): Normalizes image values to the range [0, 1] based on predefined MIN_BOUND and MAX_BOUND.
*   zero_center(image): Subtracts a predefined mean value (PIXEL_MEAN) from the image.
