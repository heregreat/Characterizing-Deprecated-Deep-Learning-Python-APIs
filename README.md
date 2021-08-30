# Characterizing Deprecated Deep Learning Python APIs: An Empirical Study on TensorFlow
# Nian Liu
### This respository consists of experiment data, script, result for this study. Below is the introduction of files in each folder. 
## Experiment Data
- models-r1.10.0-models-r2.3.0: Deep Learning projects used in our study (Models). Models range from 1.10.0 to 2.3.0, 
  there is no release for TensorFlow 1.14 and 1.15
## RQ1
- ChangeDetect.py: script used to find deprecated APIs, removed APIs, newly deprecated APIs, etc. in TensorFlow 1.0-2.3.
- output/tensorflow-r1.0.csv-tensorflow-r2.3.csv: deprecated APIs in TensorFlow 1.0-2.3.
- output/removed_apis.csv: removed APIs in TensorFlow.  
- output/age.csv: data of the number of versions deprecated APIs get deprecated since introduced into TensorFlow.
- output/api_survival_time.csv: data of the survival time of deprecated APIs get removed from TensorFlow.
- output/diff.csv: data of the newly deprecated APIs in each TensorFlow Version.
## RQ2
- extract_deprecation_message.py: script used to extract deprecation message of deprecated APIs.
- Reason for Method Deprecation.md: deprecated APIs for each method deprecation reason in RQ2.
- Reason for Parameter Deprecation.md: deprecated APIs for each parameter deprecation reason in RQ2.
## RQ3
- Compare.py: script used to find deprecated APIs in Models projects.
- output/models-r1.10.0(official).csv-models-r2.3.0(official).csv: data of deprecated APIs usage in each Model verison.
- output/models_all(official).csv: data of deprecated APIs usage in Models from version 1.10.0 to 2.3.0 without false positive.
## RQ4
- p-value.py: script used to calculate p-value and cohen'd value.
- commit_history.numbers: commit history of each commit that updated deprecated APIs.
- log: original log which contains model training and testing result for the 13 tests in RQ4.

## Other files
- utils: other files.
