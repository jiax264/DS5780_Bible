# Automated Text Simplification with AI: Fine-Tuning a Small Pretrained Transformer Model and In-Context Learning with a Large Language Model 

## Data Sources
The primary data sources for this analysis are two versions of the Bible: the King James Version (KJV) and the Bible in Basic English (BBE) found in data/bibles

- **King James Version (KJV)**: A classic English translation of the Bible, originally published in 1611. Known for its archaic language and complex structure, it is revered for its historical significance and literary style.
- **Bible in Basic English (BBE)**: A simpler version of the Bible, translated by Professor S. H. Hooke in 1941. This version uses a vocabulary of 1,000 basic English words, making it more accessible to those with limited education or for whom English is a second language.

Both KJV and BBE are organized into folders representing each book of the Bible, with separate text files for the chapters within each book.

## Purpose of Analysis
The analysis focuses on automated text simplification by comparing traditional fine-tuning of smaller transformer models with in-context learning using large language models. The primary goal is to assess whether these methods can transform the complex and archaic language of KJV into a simpler, more readable format akin to BBE. The study examines two key aspects:

- **Semantic Similarity**: The ability of the generated text to maintain semantic fidelity to the original KJV.
- **Readability Complexity**: Whether the generated text achieves lower readability complexity, similar to BBE, indicating it is easier to understand.

## Data Preprocessing
- **Data_Clean.ipynb → bible_cleaned_data.csv**: Collects and organizes chapters from King James Version (KJV) and Bible in Basic English (BBE) into a dataframe, with basic text cleaning to remove leading characters and newline inconsistencies.
- **Data_Trim.ipynb → bible_cleaned_and_short_data.csv**: Trims each chapter to the first two sentences to reduce computational burden, removing rows with misalignment between KJV and BBE and discarding chapters from Psalms due to complex headings.

## Modeling
- **finetune_mvp.ipynb → train_data.csv, test_data.csv, mvp_test_data.csv**: Splits the cleaned data into training and test sets using a random seed, and fine-tunes the Multi-task Supervised Pre-training for Natural Language Generation (MVP) model. Generates BBE-like text from KJV input and sets inference parameters.
- **Create_embeddings_and_nearest_neigh_from_train.ipynb → test_with_nearest_neigh_from_train_df.csv**: Identifies the semantically most similar KJV passage from the training set to use as the example for in-context learning with the Gemma model. Embeds KJV-BBE pairs to find the closest neighbor for guidance.
- **in_context_learning_gemma.ipynb → gemma_test_data.csv**: Uses the Google Gemma model for in-context learning to generate BBE-like text from KJV. Implements model authentication and generates text using a provided prompt. Post-processes the output to ensure consistent format.

## Postprocessing
- **Final_Merge_and_Cleaning_Before_Evaluation.ipynb → final_merged_test_df.csv**: Merges Gemma-generated and MVP-generated BBE text into a unified dataframe, removing incomplete or erroneous results. Finalizes the dataset by removing rows with significant divergence or low content value, leading to a refined dataset.

## Evaluation
- **Evaluation_Semantic_Similarity.ipynb**: Uses embeddings to calculate semantic similarity between KJV and generated texts, applying non-parametric statistical tests to compare cosine similarities. The focus is on assessing whether MVP and Gemma-generated texts are more semantically similar to KJV than the original BBE.
- **Evaluation_Readability.ipynb**: Calculates readability scores for each text using the Dale-Chall Readability Formula. Assesses readability differences between KJV, BBE, Gemma, and MVP-generated BBE, using non-parametric tests to evaluate the significance of results.
