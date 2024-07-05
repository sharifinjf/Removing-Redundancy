<p align="center">
  <img src="Images/Row.png" > <br>
  Row Images
</p>
<p align="center">
  <img src="Images/1D.png"> <br>
  One-dimension, remove redundancy 
</p>
<p align="center">
  <img src="Images/2D.png" > <br>
  Two-dimension, remove redundancy
</p>



# Abstract 
One of the most important challenges in the preparation and maintenance of databases such as face images is the presence of a large number of variables in the raw data. Reducing the size of each observation so that its features do not disappear and the co-signer uniquely provides dependence on the original data is a necessary requirement of functioning with databases. Principal component analysis and its improved methods are the common techniques for reducing the dimensionality of such datasets and simultaneously increasing interpretability and minimizing information loss. In this paper, our purpose is to define one preprocessing step as nonuniform sampling to preserve raw data appearance and to increase the performance of the dimensional reduction methods. We use the properties of sparse principal component analysis to identify the location of less important values of the raw data that do not interfere with data features. By using sparse eigenvectors, two algorithms are presented to remove redundancy from the raw data in the one-dimensional and two-dimensional cases. After removing raw data redundancy, newly obtained data in other applications such as database recognition and compression can be used. Simulation results show that using this preprocessing step reduces the memory amount and also provides a higher recognition rate.


## Getting Started

### Step 1: Preparing Your Dataset
1. Place your dataset in the `matlabe code/Remove column/data` folder.
2. Run `load_data.m` to load and preprocess the data.

### Step 2: Using One-Dimensional Algorithms
1. Navigate to the `Remove column` directory.
2. Depending on your algorithm, you must run either `SLE_remove.m` or `remove_spca.m`.

### Step 3: Using Two-Dimensional Algorithms
1. Navigate to the `Remove column and row` directory.
2. Select and run either `slec_crop.m` or `SPACA_crop.m`.

## Contributing
Feel free to contribute by submitting issues or pull requests. Your feedback and contributions are welcome!

If you find this codebase useful, please consider citing:
@article{sharifi2019removing,
  title={Removing redundancy data with preserving the structure and visuality in a database},
  author={Sharifi Najafabadi, Ali Asghar and Torkamani Azar, Farah},
  journal={Signal, Image and Video Processing},
  volume={13},
  number={4},
  pages={745--752},
  year={2019},
  publisher={Springer}
}
