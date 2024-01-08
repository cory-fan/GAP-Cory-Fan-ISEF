GitHub Repo for Cory Fan's ISEF submission.

Currently contains only classification model for GAP. Please download the HDF5 format version of ModelNet40 with 2048 sampled points, create a folder /gap_cls/data/, and unzip the ModelNet40 dataset in that folder.

The segmentation model, other GAP-augmentation models, and instructions for the Kumamoto Dataset creation will be uploaded shortly.

Thanks to the following repos for code used:

@article{ma2022rethinking,
    title={Rethinking network design and local geometry in point cloud: A simple residual MLP framework},
    author={Ma, Xu and Qin, Can and You, Haoxuan and Ran, Haoxi and Fu, Yun},
    journal={arXiv preprint arXiv:2202.07123},
    year={2022}
}

@article{Pytorch_Pointnet_Pointnet2,
      Author = {Xu Yan},
      Title = {Pointnet/Pointnet++ Pytorch},
      Journal = {https://github.com/yanx27/Pointnet_Pointnet2_pytorch},
      Year = {2019}
}
