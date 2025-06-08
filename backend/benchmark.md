# Benchmark Documentation

This document summarizes the benchmarking results for feature extraction and model training in this workspace.

---

## 1. Feature Extraction Thread Benchmark

The script [`thread_benchmark.py`](thread_benchmark.py) was used to measure the performance of parallel feature extraction with varying thread counts. The dataset used was [`datasets/BenchmarkSubset.csv`](datasets/BenchmarkSubset.csv).

### Results

| Threads | Time (seconds) |
|---------|---------------|
|   16    |    347.39     |
|   32    |    190.95     |
|   48    |    165.51     |
|   64    |    134.38     |

✅ Feature extraction completed successfully.

---

## 2. Benchmark XGBoost GPU

*Results from [`benchmark_xgb_gpu.py`](benchmark_xgb_gpu.py) will be added here once available.* *2k URLS*
 Creating default XGBoost model...
✅ Saved default metrics to models/xgb_gpu_default.csv

⚙️ Creating and tuning XGBoost with Optuna (GPU)...
      Accuracy     AUC  Recall   Prec.      F1   Kappa     MCC                                                                                                                                                                         
Fold                                                          
0       0.8571  0.8730  0.5758  0.7600  0.6552  0.5672  0.5759
1       0.8286  0.8401  0.5152  0.6800  0.5862  0.4807  0.4880
2       0.8786  0.8995  0.6667  0.7857  0.7213  0.6444  0.6479
3       0.8714  0.9000  0.6364  0.7778  0.7000  0.6192  0.6243
4       0.8714  0.9065  0.5758  0.8261  0.6786  0.6014  0.6167
5       0.8929  0.8434  0.6061  0.9091  0.7273  0.6639  0.6850
6       0.8643  0.8754  0.5758  0.7917  0.6667  0.5841  0.5958
7       0.8786  0.8865  0.6176  0.8400  0.7119  0.6372  0.6493
8       0.9071  0.8998  0.6471  0.9565  0.7719  0.7163  0.7379
9       0.9000  0.9498  0.6765  0.8846  0.7667  0.7045  0.7147
Mean    0.8750  0.8874  0.6093  0.8211  0.6986  0.6219  0.6336
Std     0.0215  0.0304  0.0472  0.0762  0.0524  0.0656  0.0685
✅ Tuning completed in 180.22 seconds.

📈 Saved tuned metrics to models/xgb_gpu_tuned.csv

🔍 Comparing Default vs. Tuned XGBoost (based on mean scores):
🔹 Accuracy: Default = 0.7895, Tuned = 0.8039 → Best: TUNED
🔹 AUC: Default = 0.8047, Tuned = 0.8160 → Best: TUNED
🔹 F1: Default = 0.6202, Tuned = 0.6447 → Best: TUNED

✅ GPU usage log saved to models/gpu_usage.log
🚀 Benchmarking complete! Review metrics and GPU logs for benchmark.md.
root@a4d389a941bb:/workspaces/BrowserExtension/backend# nvidia-smi
Sat Jun  7 22:53:30 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 575.57.04              Driver Version: 576.52         CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 2070 ...    On  |   00000000:01:00.0  On |                  N/A |
| 34%   46C    P5             26W /  210W |    1002MiB /   8192MiB |     24%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
root@a4d389a941bb:/workspaces/BrowserExtension/backend# 
---


## 3. Benchmark XGBoost GPU Full DATASET

*Results from [`train_model.py`](train_model.py) will be added here once available.* *FULL DATASET 450k URLS*

█████████████████████| 450176/450176 [2:39:46<00:00, 46.96it/s]
✅ Extracted features for 450165 URLs.
✅ Saved extracted features to models/extracted_features_full.csv

 Creating and tuning XGBoost with Optuna (GPU)...
      Accuracy     AUC  Recall   Prec.      F1   Kappa     MCC                                                                                                          
Fold                                                          
0       0.7028  0.8756  0.8644  0.4301  0.5744  0.3834  0.4387
1       0.7020  0.8730  0.8588  0.4289  0.5721  0.3803  0.4345
2       0.7006  0.8728  0.8624  0.4279  0.5720  0.3796  0.4349
3       0.7028  0.8722  0.8517  0.4292  0.5707  0.3792  0.4315
4       0.6979  0.8737  0.8642  0.4256  0.5703  0.3764  0.4328
5       0.6995  0.8769  0.8714  0.4275  0.5736  0.3809  0.4388
6       0.7030  0.8762  0.8607  0.4300  0.5735  0.3824  0.4367
7       0.7012  0.8724  0.8631  0.4285  0.5726  0.3806  0.4360
8       0.6978  0.8745  0.8632  0.4254  0.5699  0.3759  0.4321
9       0.6979  0.8718  0.8609  0.4253  0.5693  0.3753  0.4309
Mean    0.7005  0.8739  0.8621  0.4278  0.5718  0.3794  0.4347
Std     0.0020  0.0017  0.0047  0.0018  0.0016  0.0026  0.0027
✅ Tuning completed in 638.73 seconds.

📊 Saved tuned metrics to models/xgb_gpu_tuned_full.csv
Transformation Pipeline and Model Successfully Saved
✅ Saved tuned model to models/xgb_gpu_final.pkl
✅ GPU usage log saved to models/gpu_usage_full.log
🚀 Full model training complete! Ready for deployment.

## 4. Benchmark n_iter 20k sample on extracted features dataset
*Results from [`benchmark_n_iter.py`](benchmark_n_iter.py) will be added here once available.* 

n_iter,AUC,Accuracy,F1,Recall,Prec.,Time_sec,Timestamp
20,0.7833583333333335,0.6844166666666666,0.5364,0.7168833333333335,0.42910833333333337,186.59,2025-06-08T06:06:54.972476
40,0.7842416666666668,0.6466916666666666,0.5192833333333334,0.7716000000000002,0.3915416666666667,205.01,2025-06-08T06:10:22.051096
60,0.7842416666666668,0.6466916666666666,0.5192833333333334,0.7716000000000002,0.3915416666666667,235.16,2025-06-08T06:14:19.493916
80,0.7848333333333334,0.7901916666666667,0.572025,0.46024166666666666,0.7604083333333334,253.08,2025-06-08T06:18:35.001502

## 5. Benchmark scale_pos_weight 20k sample on extracted features dataset
*Results from [`benchmark_spw.py`](benchmark_spw.py) will be added here once available.* 

Benchmarking completed successfully!
📊 Summary of results:
   scale_pos_weight       AUC  Accuracy        F1    Recall     Prec.  Time_sec                   Timestamp
0              1.00  0.784833  0.790192  0.572025  0.460242  0.760408    123.44  2025-06-08T06:32:45.070468
1              2.00  0.784833  0.790192  0.572025  0.460242  0.760408    123.80  2025-06-08T06:34:51.260582
2              3.31  0.784833  0.790192  0.572025  0.460242  0.760408    125.11  2025-06-08T06:36:58.847888
3              5.00  0.784833  0.790192  0.572025  0.460242  0.760408    123.70  2025-06-08T06:39:05.031198
4              7.33  0.784833  0.790192  0.572025  0.460242  0.760408    123.31  2025-06-08T06:41:10.824076
5             10.00  0.784833  0.790192  0.572025  0.460242  0.760408    123.34  2025-06-08T06:43:16.631589
6             15.00  0.784833  0.790192  0.572025  0.460242  0.760408    124.09  2025-06-08T06:45:23.199281


## Inference metrics Summary (22k dataset)
*Results from [`inference_variants_sample.py`](inference_variants_sample.py) will be added here once available.* 

Model,AUC,Recall,Precision,F1,Time_sec,CPU_%,Memory_%
default,0.87613,0.8684,0.4261,0.5717,0.29,0.4,27.1
baseline,0.875374,0.7252,0.5798,0.6444,0.11,0.9,27.3
spw_3.31,0.875374,0.7252,0.5798,0.6444,0.11,0.0,27.3
spw_3.31_dropped,0.874338,0.7197,0.5811,0.643,0.11,0.4,27.3
