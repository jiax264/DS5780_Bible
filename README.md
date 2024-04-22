# Automated Text Simplification with AI: Fine-Tuning a Small Pretrained Transformer Model and In-Context Learning with a Large Language Model 

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
