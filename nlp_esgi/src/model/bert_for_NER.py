from transformers import BertTokenizer, BertForTokenClassification
from transformers import AdamW
import torch


class BertForNERModel:
    """
    This class represents a BERT model fine-tuned for Named Entity Recognition.
    """

    def __init__(self):
        # Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        # Initialize the BERT model for token classification
        self.model = BertForTokenClassification.from_pretrained(
            'bert-base-cased',
            num_labels=3  # Number of distinct labels in your NER task, e.g., O, B-PER, I-PER
        )

    def encode_data(self, texts):
        """Encodes the data into the format expected by BERT."""
        input_ids = []
        attention_masks = []

        for text in texts:
            encoded_dict = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=64,  # Pad & truncate all sentences.
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',  # Return pytorch tensors.
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

    def fit(self, texts, labels):
        """Fine-tune the BERT model."""
        input_ids, attention_masks = self.encode_data(texts)

        # Convert label ids to tensors
        labels = torch.tensor(labels)

        # Create the DataLoader for our training set
        train_data = TensorDataset(input_ids, attention_masks, labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=32)

        # Set up the optimizer
        optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)

        # Train loop...
        for epoch in range(4):  # Recommended training for 3-4 epochs
            # Training step
            self.model.train()
            # Loop over batches
            for step, batch in enumerate(train_dataloader):
                b_input_ids, b_input_mask, b_labels = batch
                self.model.zero_grad()
                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

    def predict(self, texts):
        """Predict entities in the given text."""
        input_ids, attention_masks = self.encode_data(texts)

        # Prediction step
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, token_type_ids=None, attention_mask=attention_masks)
            logits = outputs.logits

        predictions = torch.argmax(logits, dim=2)
        return predictions.numpy()  # Convert predictions to numpy array

    def dump(self, filename_output):
        """Serialize and save the model to the specified file."""
        torch.save(self.model.state_dict(), filename_output)

    def load(self, filename_input):
        """Load and deserialize the model from the specified file."""
        self.model.load_state_dict(torch.load(filename_input))

