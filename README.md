# Adaptive Statistical Profile (ASP) for Keystroke Authentication

## Overview

This repository contains the implementation of the **Adaptive Statistical Profile (ASP)** method for keystroke dynamics authentication, as described in the academic paper:

> **"Session-Adaptive Keystroke Dynamics: Outlier-Resilient Profiling with Robust Statistics"**  
> Author: Yuji Umemoto (Kwassui Women's University), Ken-ichi Tanaka (Nagasaki Institute of Applied Science)  
> *Submitted to: Journal of the Institute of Image Information and Television Engineers* (under review)

## Key Features

- **Robust Statistics**: Uses median, IQR, and MAD instead of mean and standard deviation for outlier resistance
- **Adaptive Weighting**: Dynamically adjusts feature importance based on stability and reliability
- **Multi-Metric Distance**: Combines three robust distance measures with weighted coefficients
- **High Performance**: Achieves 11.97% EER on the CMU Keystroke Dynamics Dataset

## Performance Results

| Method | EER (%) | Improvement |
|--------|---------|-------------|
| **ASP (Proposed)** | **11.97** | **-** |
| Scaled Manhattan | 12.98 | 7.8% better |
| Manhattan Distance | 19.57 | 38.8% better |
| Euclidean Distance | 21.53 | 44.4% better |

## Requirements

- Python 3.7+
- pandas
- numpy
- CMU Keystroke Dynamics Dataset

## Installation

1. Clone this repository:
```bash
git clone https://github.com/[your-username]/asp-keystroke-authentication.git
cd asp-keystroke-authentication
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the CMU Keystroke Dynamics Dataset:
   - Visit: https://www.cs.cmu.edu/~keystroke/
   - Download `DSL-StrongPasswordData.csv`
   - Place it in the project directory

## Usage

### Basic Usage

```python
from asp_simple_implementation import ASPKeystrokeAuthenticator

# Initialize the authenticator
asp = ASPKeystrokeAuthenticator('DSL-StrongPasswordData.csv')

# Run evaluation
results = asp.evaluate_performance()
print(f"EER: {results['eer']*100:.2f}%")
```

### Command Line Usage

```bash
python asp_simple_implementation.py
```

## Algorithm Details

### 1. Robust Profile Creation
- **Median**: Outlier-resistant central tendency
- **IQR**: Interquartile Range for robust scale
- **MAD**: Median Absolute Deviation for variability

### 2. Adaptive Weighting
- **Stability Weight**: `w_s = 1/(1 + CV)` where CV is coefficient of variation
- **Reliability Weight**: `w_r = min(1.0, √n/10.0)` where n is sample count
- **Composite Weight**: `w_c = w_s × w_r`

### 3. Multi-Metric Distance
```
ASP_distance = 0.5 × median_dist + 0.3 × mad_dist + 0.2 × std_dist
```

## Dataset Information

The implementation uses the CMU Keystroke Dynamics Dataset:
- **Users**: 51 participants
- **Samples**: 20,400 keystroke samples
- **Features**: 31 timing features (dwell time, flight time)
- **Password**: ".tie5Roanl" (fixed password)

## File Structure

```
asp-keystroke-authentication/
├── README.md                    # This file
├── asp_simple_implementation.py # Main implementation
├── requirements.txt             # Python dependencies
├── sample_output.txt           # Example output
└── LICENSE                     # License file
```

## Reproducibility

All results are reproducible using:
- Fixed random seed (42)
- Deterministic train-test split
- Consistent feature ordering
- Exact algorithm implementation as described in the paper

## Citation

If you use this code in your research, please cite:

```bibtex
@article{umemoto2025asp,
  title={Session-Adaptive Keystroke Dynamics: Outlier-Resilient Profiling with Robust Statistics},
  author={Umemoto, Yuji and Tanaka, Ken-ichi},
  journal={Journal of the Institute of Image Information and Television Engineers},
  note={Submitted for publication},
  year={2025}
}
```

## Author

**Yuji Umemoto**  
Assistant Professor  
Faculty of Contemporary Culture  
Kwassui Women's University  
Nagasaki, Japan

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- CMU Keystroke Dynamics Dataset creators: Kevin S. Killourhy and Roy A. Maxion
- Support from Kwassui Women's University and Nagasaki Institute of Applied Science