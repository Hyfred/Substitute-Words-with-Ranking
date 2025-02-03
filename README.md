# **Learning to Substitute Words with Model-based Score Ranking**  

This is the repository for paper: Learning to Substitute Words with Model-based Score Ranking

## **Dataset**  
The dataset is based on [Microsoft's SmartWordSuggestions](https://github.com/microsoft/SmartWordSuggestions).  

## **Training**  
We provide two training methods for fine-tuning a BERT-based model:  

1. **DPO-based Training**  
   - Fine-tunes the model using **DPO** / **σ-DPO** loss.  
   - Run the following command:  
     ```
     cd dpo_based_training
     python run.py
     ```  

2. **Margin Ranking-based Training**  
   - Fine-tunes the model using **MR / MR-AS / MR-BS** loss.  
   - Run the following command:  
     ```
     cd marginranking_based_training
     python run.py
     ```  

## **Evaluation**  
- **p-value Calculation**  
  - Computes the **p-value metric** to evaluate model performance.  
     ```
     cd p_value
     ```  

## **Inference**  
- **Generating Word Substitutions**  
  - Uses a trained model to generate substitution candidates.  
  - Pretrained model weights can be downloaded [here](https://drive.google.com/file/d/1wzsqwfac9S25dEqu9xxJlRIvQleoAEKo/view?usp=sharing).  
  - Run the following command to generate the **top-2 candidate words** for a given input:  
    ```
    cd inference
    python inference.py
    ```  
  - Modify the script as needed to generate more candidates.  