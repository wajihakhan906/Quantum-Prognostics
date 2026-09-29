# Datasets

The raw vibration data is not redistributed here. Download it and place it in this folder.

## Fault diagnosis: CWRU Bearing Data Center
Case Western Reserve University, 12 kHz drive-end accelerometer, 4 health states.
https://engineering.case.edu/bearingdatacenter/download-data-file

```
Dataset/CWRU/
├── Normal/      97.mat, 98.mat, ...
├── Ball/        118.mat, ...
├── InnerRace/   105.mat, ...
└── OuterRace/   130.mat, ...
```
`Code/data.py::load_cwru` reads the `*_DE_time` signal from each file and cuts it into 2048-sample windows (50 % overlap).

## Remaining useful life: XJTU-SY run-to-failure bearings
Xi'an Jiaotong University & Changxing Sumyoung Technology, 25.6 kHz, 3 operating conditions, 15 bearings.
https://biaowang.tech/xjtu-sy-bearing-datasets/

```
Dataset/XJTU-SY/35Hz12kN/Bearing1_1/1.csv ... N.csv
```
Each CSV is a 1.28 s snapshot taken every minute; RUL is labelled linearly from 1 (new) to 0 (failure).

## Synthetic mode
`--synthetic` generates toy impulse-train signals so the full pipeline can be smoke-tested without downloads.
Synthetic numbers are **not** research results.
