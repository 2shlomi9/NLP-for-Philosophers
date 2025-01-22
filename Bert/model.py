import torch
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

def initialize_model(train_loader, epochs=4):
    config = BertConfig.from_pretrained(
        'bert-base-uncased',
        num_labels=4,  # Number of classes in dataset
        hidden_dropout_prob=0.44,  # Dropout probability for hidden layers
        attention_probs_dropout_prob=0.44  # Dropout probability for attention layers
    )

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=3e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    return model, tokenizer, optimizer, scheduler, device