# Training for Verification (T4V)
This is the repository for [T4V: Exploring Neural Network Architectures that Improve the Scalability of Neural Network Verification](https://link.springer.com/chapter/10.1007/978-3-031-22337-2_28), which focuses on improving the scalability of NN verification through exploring which NN architectures lead to more scalable verification. In this paper, we propose a general framework for incorporating verification scalability in the training process by identifying NN properties that improve verification and incentivizing these properties through a verification loss. One natural application of our method is robustness verification, especially using tools based on interval analysis, which have shown great promise in recent years. Specifically, we show that we can greatly reduce the approximation error of interval analysis by forcing all (or most) NNs to have the same sign. Finally, we provide an extensive evaluation on the MNIST and CIFAR-10 datasets in order to illustrate the benefit of training for verification. This repository contains the code necesary to replicate our experiments.

## Prerequisites
1. Clone the [https://github.com/KaidiXu/LiRPA_Verify](LiRPA_Verify) (aka Fast and Complete) repository into the `src` directory.
2. Follow the installation and setup instructions for LiRPA_Verify.
3. Run our setup script.
   ```
   ./setup_fac.sh
   ```

## MNIST Experiments
### Training for Scenario 1 (Classification Accuracy)
In this scenario, we apply our verification loss as a penalty to the cross-entropy loss, which incentivizes classification accuracy. To replicate our results for MNIST Scenario 1, first train the fully-connected neural networks from the `src` directory.
```
python3 train_FC.py --num_hidden_layers [NUM HIDDEN LAYERS] --layer_size [LAYER SIZE] --epochs [EPOCHS] --warmup_epochs [WARMUP EPOCHS] --asymmetric_l2 --gamma 0.9 --alpha [ALPHA] --seed [SEED]
```
In our experiments for Scenario 1, we use the following hyperparameters (54 unique networks). Pretrained models can be found in `src/mnist_models`.
| Architecture | NUM HIDDEN LAYERS | LAYER SIZE | EPOCHS | WARMUP EPOCHS | ALPHA                              | SEED    |
| ---          | ---               | ---        | ---    | ---           | ---                                | ---     | 
| 2x50         | 2                 | 50         | 120    | 30            | 0.70, 0.76, 0.82, 0.88, 0.94, 1.00 | 0, 1, 2 |
| 2x200        | 2                 | 200        | 100    | 10            | 0.50, 0.60, 0.70, 0.80, 0.90, 1.00 | 0, 1, 2 |
| 5x200        | 5                 | 200        | 100    | 10            | 0.65, 0.72, 0.79, 0.86, 0.93, 1.00 | 0, 1, 2 |

### Training for Scenario 2 (Classification Accuracy and Robustness)
In this scenario, we apply our verification loss as a penalty to the combined cross-entropy and interval bound propagation loss, which incentivizes both classification accuracy and robustness. To replicate our results for MNIST Scenario 2, first train the fully-connected neural networks from the `src` directory.
```
python3 train_FC.py --num_hidden_layers [NUM HIDDEN LAYERS] --layer_size [LAYER SIZE] --epochs [EPOCHS] --warmup_epochs [WARMUP EPOCHS] --asymmetric_l2 --interval_prop --interval_width 0.04 --gamma 0.9 --alpha [ALPHA] --beta [BETA] --seed [SEED]
```
In our experiments for Scenario 2, we use the following hyperparameters (54 unique networks). Pretrained models can be found in `src/mnist_models`.
| Architecture | NUM HIDDEN LAYERS | LAYER SIZE | EPOCHS | WARMUP EPOCHS | ALPHA                              | BETA | SEED    |
| ---          | ---               | ---        | ---    | ---           | ---                                | ---  | ---     | 
| 2x50         | 2                 | 50         | 120    | 30            | 0.90, 0.92, 0.94, 0.96, 0.98, 1.00 | 0.84 | 0, 1, 2 |
| 2x200        | 2                 | 200        | 100    | 10            | 0.75, 0.80, 0.85, 0.90, 0.95, 1.00 | 0.96 | 0, 1, 2 |
| 5x200        | 5                 | 200        | 100    | 10            | 0.95, 0.96, 0.97, 0.98, 0.99, 1.00 | 0.97 | 0, 1, 2 |

### Verification for Scenarios 1 and 2
To verify the trained networks, run:
```
./verify_FC.sh [MODEL PATH] [NUM HIDDEN LAYERS] [LAYER SIZE] [EPSILON] [OUTPUT LOG PATH]
```
For example, to verify a pretrained 2x50 Scenario 1 network with $\alpha=0.70$, run:
```
./verify_FC.sh mnist_models/2x50_asymL2_alpha0-7_gamma0-9_seed0.pth 2 50 20 2x50_asymL2_alpha0-7_gamma0-9_seed0_log.txt
```
In our experiments we use the following values of EPSILON.
| Architecture | Scenario 1 EPSILON | Scenario 2 EPSILON |
| ---          | ---                | ---                |
| 2x50         | 20                 | 25                 |
| 2x200        | 15                 | 20                 |
| 5x200        | 8                  | 15                 |

Summarize the results in the log file with the `summarize_results.py` script. For example, to summarize results from a single trial recorded in `log_file.txt`, run:
```
python3 summarize_results.py log_file.txt
```
To summarize results and average over multiple trials (e.g., seeds 0, 1, and 2), run:
```
python3 summarize_results.py log_file_a.txt,log_file_b.txt,log_file_c.txt
```

### Robustness for Scenarios 1 and 2
To evaluate the robustness of the trained MNIST networks, we apply the PGD attack. For Scenario 1, run
```
python3 pgd_FC.py --load [MODEL PATH] --seed [SEED] --eps [EPSILON] --hidden_layers [NUM HIDDEN LAYERS] --layer_size [LAYER SIZE] --asymmetric_l2 --alpha [ALPHA] --gamma 0.9
```
and for Scenario 2, run
```
python3 pgd_FC.py --load [MODEL PATH] --seed [SEED] --eps [EPSILON] --hidden_layers [NUM HIDDEN LAYERS] --layer_size [LAYER SIZE] --asymmetric_l2 --alpha [ALPHA] --gamma 0.9 --interval_prop --interval_width 0.04 --beta [BETA]
```
For each network, we use the same EPSILON for evaluating robustness as the one we use for verification. For the remaining hyperparameters, supply those that the network was trained with.

## CIFAR-10 Experiments
### Training for Scenario 1 (Classification Accuracy)
In this scenario, we apply our verification loss as a penalty to the cross-entropy loss, which incentivizes classification accuracy. To replicate our results for CIFAR-10 Scenario 1, first train convolutional neural networks from the `src` directory.
```
python3 train_CNN.py --asymmetric_l2 --gamma 0.9 --alpha [ALPHA] --seed [SEED]
```
In our experiments for Scenario 1, we use the hyperparameters ALPHA $\in$ {0.80, 0.85, 0.90, 0.95, 0.99, 1.00} and SEED $\in$ {0, 1, 2}, for 18 unique networks. Pretrained models can be found in `src/cifar10_models`.

### Training for Scenario 2 (Classification Accuracy and Robustness)
In this scenario, we apply our verification loss as a penalty to the combined cross-entropy and interval bound propagation loss, which incentivizes both classification accuracy and robustness. To replicate our results for CIFAR-10 Scenario 2, first train the convolutional neural networks from the `src` directory.
```
python3 train_CNN.py --asymmetric_l2 --interval_prop --gamma 0.9 --alpha [ALPHA] --beta 0.995 --seed [SEED]
```
In our experiments for Scenario 2, we use the hyperparameters ALPHA $\in$ {0.85, 0.88, 0.91, 0.94, 0.97, 1.00} and SEED $\in$ {0, 1, 2}, for 18 unique networks. Pretrained models can be found in `src/cifar10_models`.

### Verification for Scenarios 1 and 2
To verify the trained networks, run:
```
./verify_CNN.sh [MODEL PATH] [EPSILON] [OUTPUT LOG PATH]
```
For example, to verify a pretrained Scenario 1 network with $\alpha=0.80$, run:
```
./verify_CNN.sh cifar10_models/CNN_model_0_asymL2_alpha_0.8_gamma_0.9_seed0.pth 3 CNN_model_0_asymL2_alpha_0.8_gamma_0.9_seed0_log.txt
```
In our experiments we use EPSILON=3 for both Scenario 1 and Scenario 2.

Results for the CIFAR-10 experiments can also be summarized using `summarize_results.py` script.

### Robustness for Scenarios 1 and 2
To evaluate the robustness of the trained CIFAR-10 networks, we apply the PGD attack. For Scenario 1, run
```
python3 pgd_CNN.py --load [MODEL PATH] --seed [SEED] --eps [EPSILON] --asymmetric_l2 --alpha [ALPHA] --gamma 0.9
```
and for Scenario 2, run
```
python3 pgd_CNN.py --load [MODEL PATH] --seed [SEED] --eps [EPSILON] --asymmetric_l2 --alpha [ALPHA] --gamma 0.9 --interval_prop --interval_width 0.04 --beta [BETA]
```
For each network, we use the same EPSILON for evaluating robustness as the one we use for verification. For the remaining hyperparameters, supply those that the network was trained with.
