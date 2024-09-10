This directory contains the code for predicting the difficulty of code snippets in our study. There are two subdirectories:

- The first directory is for model input with 10%-50% few-reliance simulations.
- The second directory is for model input with 60%-100% few-reliance simulations and others.

To run the model, use the following command:

```bash
python stim_evaluation.py --problem-setting=subjective_difficulty --split=code-snippet --mode=bimodal --simulation="simulation type" --seed=41 --output="output.csv"
```



The final_result.csv file contains the four metrics for model performance with different scanpath inputs. Since we conducted 3-fold cross-validation, there are three lines for each model input, representing the result after each cross-validation.

Due to the space limitations of GitHub, the input embeddings are stored in an anonymous Dropbox account. The link is provided here: https://www.dropbox.com/scl/fo/rpf389602ek8ugaujq12u/ANXytQ-AsY_oK7xJiO5gWR4?rlkey=vucxhwkl82mh2xened5rn67n2&st=dpiv1jcy&dl=0
