# QuickNeRF
A reimplementation of Hash Encoding in InstantNGP for NeRF with quick simplified intro to NeRF

### Experiment Table

Experiment with RTX 2060 Max-Q

|                        | Model size | Step | PSNR  | SSIM  | LPIPS |
| ---------------------- | ---------- | ---- | ----- | ----- | ----- |
| LEGO baseline          | 1192k      | 50k  | 28.73 | 0.937 | 0.073 |
| LEGO hash              | 19.2k      | 10k  | 28.45 | 0.940 | 0.068 |
| MIC baseline           | 1192k      | 50k  | 30.53 | 0.967 | 0.051 |
| MIC hash               | 19.2k      | 10k  | 28.19 | 0.960 | 0.055 |
| SHIP baseline          | 1192k      | 50k  | 27.34 | 0.834 | 0.194 |
| SHIP hash              | 19.2k      | 10k  | 26.37 | 0.834 | 0.175 |
| (llff) ORCHID hash     | 19.2k      | 10k  | 17.58 | 0.688 | 0.227 |
| (llff) ORCHID baseline | 1192k      | 50k  | 20.57 | 0.686 | 0.297 |

### T experiment

| T    | PSNR  | SSIM  | LPIPS |
| ---- | ----- | ----- | ----- |
| 13   | 25.87 | 0.878 | 0.141 |
| 16   | 27.55 | 0.921 | 0.094 |
| 19   | 28.45 | 0.940 | 0.067 |
| 21   | 28.55 | 0.943 | 0.061 |

# Citation
Kudos to the authors for their amazing results and code implementations:

@article{mueller2022instant,
    author = {Thomas M\"uller and Alex Evans and Christoph Schied and Alexander Keller},
    title = {Instant Neural Graphics Primitives with a Multiresolution Hash Encoding},
    journal = {ACM Trans. Graph.},
    issue_date = {July 2022},
    volume = {41},
    number = {4},
    month = jul,
    year = {2022},
    pages = {102:1--102:15},
    articleno = {102},
    numpages = {15},
    url = {https://doi.org/10.1145/3528223.3530127},
    doi = {10.1145/3528223.3530127},
    publisher = {ACM},
    address = {New York, NY, USA},
}

@misc{mildenhall2020nerf,
    title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
    author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
    year={2020},
    eprint={2003.08934},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@misc{lin2020nerfpytorch,
  title={NeRF-pytorch},
  author={Yen-Chen, Lin},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/yenchenlin/nerf-pytorch/}},
  year={2020}
}
