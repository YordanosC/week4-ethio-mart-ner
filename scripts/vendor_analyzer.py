import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import re # Import re for regular expressions
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

class VendorAnalyzer:
    """
    Analyzes Telegram channel data to compute vendor performance metrics and a lending score.
    Requires a fine-tuned NER model to extract product, price, and location entities.
    """
    def __init__(self, data_path, model_path, tokenizer_path, id2label_map):
        self.data_path = data_path
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.id2label = id2label_map # Mapping from label ID to string label
        self.model = None
        self.tokenizer = None
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """Loads the fine-tuned NER model and tokenizer."""
        print(f"Loading model from {self.model_path} and tokenizer from {self.tokenizer_path}...")
        try:
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            self.model.eval() # Set model to evaluation mode
            print("Model and tokenizer loaded successfully.")
        except Exception as e:
            print(f"Error loading model or tokenizer: {e}")
            print("Please ensure the model and tokenizer are saved correctly and paths are accurate.")
            self.model = None
            self.tokenizer = None

    def _extract_entities(self, text):
        """
        Extracts entities (Product, Price, Location) from text using the NER model.
        Handles subword tokens and reconstructs words.
        """
        if not self.model or not self.tokenizer or not isinstance(text, str):
            return []

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        
        # Ensure model is on CPU if no GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
        
        # Get word IDs to align tokens with original words
        word_ids = inputs.word_ids(batch_index=0)

        extracted_entities = []
        current_entity = {"type": None, "text": []}
        
        # Iterate through tokens and their predicted labels
        previous_word_idx = None
        for i, pred_id in enumerate(predictions):
            # Skip special tokens (CLS, SEP, PAD)
            if word_ids[i] is None:
                continue

            current_word_idx = word_ids[i]
            token = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0, i].item())
            label = self.id2label.get(pred_id, "O") # Get label string from id

            # Handle subword tokens (e.g., ##word)
            if token.startswith("##"):
                token = token[2:] # Remove ##

            if current_word_idx != previous_word_idx: # New word
                if current_entity["type"]: # Save previous entity if exists
                    extracted_entities.append({"type": current_entity["type"], "text": "".join(current_entity["text"])})
                
                if label.startswith("B-"):
                    current_entity = {"type": label[2:], "text": [token]}
                elif label.startswith("I-"): # Should ideally not start with I-
                    current_entity = {"type": label[2:], "text": [token]} # Treat as B-
                else: # "O"
                    current_entity = {"type": None, "text": []}
            else: # Same word (subword token)
                if label.startswith("I-") and current_entity["type"] == label[2:]:
                    current_entity["text"].append(token)
                elif label.startswith("B-"): # New B- inside same word (error or complex case)
                    if current_entity["type"]:
                        extracted_entities.append({"type": current_entity["type"], "text": "".join(current_entity["text"])})
                    current_entity = {"type": label[2:], "text": [token]}
                else: # Mismatch or O for subword
                    if current_entity["type"]:
                        extracted_entities.append({"type": current_entity["type"], "text": "".join(current_entity["text"])})
                    current_entity = {"type": None, "text": []}
            
            previous_word_idx = current_word_idx
        
        if current_entity["type"]: # Save last entity if exists
            extracted_entities.append({"type": current_entity["type"], "text": "".join(current_entity["text"])})

        return extracted_entities

    def analyze_vendors(self):
        """
        Performs vendor analysis and calculates lending scores.
        Assumes 'telegram_data11.csv' has 'Message', 'Date', and 'Views' columns.
        NOTE: The 'Views' column is not typically scraped by the provided `telegram_scrapper.py`.
        For this analysis to work, you would need to modify the scraper to fetch view counts,
        or ensure your `telegram_data11.csv` includes a 'Views' column (simulated here for demonstration).
        """
        try:
            df = pd.read_csv(self.data_path)
            df.dropna(subset=['Message', 'Date'], inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])

            # Simulate 'Views' if not present in the CSV for demonstration
            if 'Views' not in df.columns:
                print("Warning: 'Views' column not found in data. Simulating views for demonstration purposes.")
                np.random.seed(42) # for reproducibility
                df['Views'] = np.random.randint(100, 5000, size=len(df)) # Random views between 100 and 5000

            vendor_metrics = {}

            for channel_username in df['Channel Username'].unique():
                vendor_df = df[df['Channel Username'] == channel_username].copy()
                
                # 1. Posting Frequency (Posts per week)
                posting_frequency = 0
                if not vendor_df.empty:
                    min_date = vendor_df['Date'].min()
                    max_date = vendor_df['Date'].max()
                    time_span_days = (max_date - min_date).days
                    if time_span_days == 0: # Handle case with only one day of data
                        time_span_weeks = 1 / 7 # Treat as one week for frequency calculation
                    else:
                        time_span_weeks = time_span_days / 7
                    
                    posting_frequency = len(vendor_df) / time_span_weeks if time_span_weeks > 0 else 0

                # 2. Average Views per Post
                average_views_per_post = vendor_df['Views'].mean() if not vendor_df.empty else 0

                # 3. Top Performing Post
                top_post_product = "N/A"
                top_post_price = "N/A"
                if not vendor_df.empty:
                    top_post = vendor_df.loc[vendor_df['Views'].idxmax()]
                    extracted_entities = self._extract_entities(top_post['Message'])
                    
                    products = [e['text'] for e in extracted_entities if e['type'] == 'PRODUCT']
                    prices = [e['text'] for e in extracted_entities if e['type'] == 'PRICE']
                    
                    top_post_product = ", ".join(products) if products else "N/A"
                    top_post_price = ", ".join(prices) if prices else "N/A"

                # 4. Average Price Point
                all_numeric_prices = []
                for message in vendor_df['Message']:
                    extracted_entities = self._extract_entities(message)
                    for entity in extracted_entities:
                        if entity['type'] == 'PRICE':
                            # Attempt to extract numeric value from price string
                            price_str = entity['text']
                            numeric_values = re.findall(r'\d+', price_str)
                            if numeric_values:
                                all_numeric_prices.extend([float(p) for p in numeric_values])
                
                average_price_point = np.mean(all_numeric_prices) if all_numeric_prices else 0

                # 5. Lending Score (example formula)
                # This is a simple weighted sum. You might need to normalize or scale these metrics
                # if their ranges are vastly different, or if you want a score between 0-100.
                # For demonstration, using raw values with arbitrary weights.
                lending_score = (average_views_per_post * 0.001) + (posting_frequency * 10) + (average_price_point * 0.01) 
                # Adjust weights (0.001, 10, 0.01) based on business importance and typical value ranges.

                vendor_metrics[channel_username] = {
                    'Avg. Views/Post': average_views_per_post,
                    'Posts/Week': posting_frequency,
                    'Avg. Price (ETB)': average_price_point,
                    'Top Post Product': top_post_product,
                    'Top Post Price': top_post_price,
                    'Lending Score': lending_score
                }
            
            return pd.DataFrame.from_dict(vendor_metrics, orient='index')

        except FileNotFoundError:
            print(f"Error: Data file not found at {self.data_path}")
            return pd.DataFrame()
        except Exception as e:
            print(f"An error occurred during vendor analysis: {e}")
            return pd.DataFrame()

# Example usage (for testing this script independently)
if __name__ == "__main__":
    # This block will only run if the script is executed directly
    print("Running VendorAnalyzer in standalone mode for testing.")
    # Dummy id2label for testing if model is not fully trained/saved yet
    # In a real scenario, this should match the id2label from your fine-tuning.
    dummy_id2label = {0: 'O', 1: 'B-PRODUCT', 2: 'I-PRODUCT', 3: 'B-PRICE', 4: 'I-PRICE', 5: 'B-LOCATION', 6: 'I-LOCATION'}
    
    # Ensure 'best_model' directory exists and contains dummy model/tokenizer files for testing
    # or point to a real saved model.
    model_dir = "./best_model" # This should be the path where your best model was saved
    
    # Create dummy files if they don't exist for standalone testing
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        # Create dummy config.json and vocab.json for AutoTokenizer/AutoModel to load
        with open(os.path.join(model_dir, 'config.json'), 'w') as f:
            f.write('{"architectures": ["XLMRobertaForTokenClassification"], "num_labels": 7}')
        with open(os.path.join(model_dir, 'vocab.json'), 'w') as f:
            f.write('{"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3, "[MASK]": 4}')
        with open(os.path.join(model_dir, 'tokenizer.json'), 'w') as f:
            f.write('{}') # Empty tokenizer.json for basic loading
        print(f"Created dummy model directory and files at {model_dir} for testing.")

    # Create a dummy telegram_data11.csv for testing
    dummy_data_path = '../data/telegram_data11.csv'
    if not os.path.exists('../data'):
        os.makedirs('../data')
    if not os.path.exists(dummy_data_path):
        dummy_data = {
            'Channel Title': ['Fashion Store A', 'Fashion Store A', 'Tech Gadgets B', 'Tech Gadgets B'],
            'Channel Username': ['@fashionA', '@fashionA', '@techB', '@techB'],
            'ID': [1, 2, 3, 4],
            'Message': [
                'አዲስ ምርት: ቆንጆ ቀሚስ ዋጋ 500 ብር.',
                'ቦሌ ላይ የሚገኝ ሱቅ. አዲስ ስቶቭ በ 1200 ብር.',
                'አዲስ ስልክ ዋጋ 15000 ብር. አዲስ አበባ.',
                'ላፕቶፕ በ 25000 ብር. ነፃ ማድረስ.'
            ],
            'Date': ['2025-06-20', '2025-06-21', '2025-06-20', '2025-06-22'],
            'Media Path': ['path1', 'path2', 'path3', 'path4']
        }
        pd.DataFrame(dummy_data).to_csv(dummy_data_path, index=False)
        print(f"Created dummy data file at {dummy_data_path} for testing.")

    analyzer = VendorAnalyzer(
        data_path=dummy_data_path,
        model_path=model_dir,
        tokenizer_path=model_dir,
        id2label_map=dummy_id2label
    )
    scorecard = analyzer.analyze_vendors()
    print("\n--- FinTech Vendor Scorecard (Standalone Test) ---")
    print(scorecard)

