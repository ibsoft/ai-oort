import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.optim import Adam
import matplotlib.pyplot as plt
import torchviz
from torch.autograd import Variable

# Function to load dataset from file


def load_dataset_from_file(file_path):
    with open(file_path, 'r') as file:
        dataset = json.load(file)
    return dataset


# Load dataset
file_path = "dataset.json"  # Update with the correct path to your dataset.json file
dataset = load_dataset_from_file(file_path)

# Check device availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Extract "request" and "response" from the dataset
requests = [item.get("request", "") for item in dataset]
responses = [item.get("response", "") for item in dataset]
greetings = [item.get("greeting", "") for item in dataset]

# Lowercase the requests and responses
requests = [request.lower() for request in requests]
responses = [response.lower() for response in responses]
greetings = [greeting.lower() for greeting in greetings]

# Define the tokenizer
tokenizer = get_tokenizer('basic_english')

# Tokenize the requests and responses
tokenized_requests = [tokenizer(request) for request in requests]
tokenized_responses = [tokenizer(response) for response in responses]
tokenized_greetings = [tokenizer(greeting) for greeting in greetings]

# Combine tokenized requests and responses
tokenized_text = tokenized_requests + tokenized_responses + tokenized_greetings

# Build the vocabulary from the tokenized text
vocab = build_vocab_from_iterator(tokenized_text)

# Add a special token for unknown words
UNK_TOKEN = "<unk>"
vocab.insert_token(UNK_TOKEN, 0)

# Replace unknown tokens in the tokenized text with the index of the unknown token
tokenized_text = [
    [token if token in vocab else UNK_TOKEN for token in tokens] for tokens in tokenized_text]

# Numericalize the text
numericalized_text = [[vocab[token] for token in tokens]
                      for tokens in tokenized_text]

# Define the dataset


class QADataset(Dataset):
    def __init__(self, data, max_length):
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        padded_item = torch.zeros(self.max_length, dtype=torch.long)
        padded_item[:len(item)] = torch.tensor(item[:self.max_length])
        return padded_item


# Load hyperparameters from config file
with open('config.json', 'r') as f:
    config = json.load(f)

model_params = config['model']
training_params = config['training']
generation_params = config['generation']

# Define the maximum length of sequences
max_sequence_length = training_params['max_sequence_length']

# Create the dataset and dataloader
dataset = QADataset(numericalized_text, max_length=max_sequence_length)
dataloader = DataLoader(
    dataset, batch_size=training_params['batch_size'], shuffle=True)

# Define the transformer-based model


class QAModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_heads, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size,
            dropout=dropout,
        )
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output = self.transformer(embedded, embedded)
        output = self.fc(output)
        return output


# Initialize the model and the optimizer
model = QAModel(
    vocab_size=len(vocab),
    embed_size=model_params['embed_size'],
    hidden_size=model_params['hidden_size'],
    num_layers=model_params['num_layers'],
    num_heads=model_params['num_heads'],
    dropout=model_params['dropout']
).to(device)

# If there are multiple GPUs, wrap the model with nn.DataParallel
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model = model.to(device)

optimizer = Adam(model.parameters(), lr=training_params['lr'])

# Train the model
losses = []  # to store loss values
for epoch in range(training_params['num_epochs']):
    epoch_losses = []  # to store losses for each epoch
    for batch in dataloader:
        x = batch.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = nn.functional.cross_entropy(
            y_pred.view(-1, len(vocab)), x.view(-1))
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())  # append loss for each batch
    # average loss for the epoch
    epoch_loss = sum(epoch_losses) / len(epoch_losses)
    losses.append(epoch_loss)  # append average loss for the epoch
    print(f'Epoch {epoch}, Loss {epoch_loss}')
    if epoch_loss < training_params['stop_loss']:
        break

print("Model trained.")

# Plot the training loss
plt.figure(figsize=(8, 6))
plt.plot(losses, marker='o', linestyle='-')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('training_loss_plot.png')  # Save the plot as an image file
plt.show()

# Save the model to disk
torch.save(model.state_dict(), 'model.pth')
print("Model saved.")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')
print(f'The model has {len(vocab)} tokens')

# Create a variable with the size of your input
x = torch.randint(high=len(vocab), size=(1, 30), dtype=torch.long).to(device)

# Generate a diagram for a specific model
y = model(x)
dot = torchviz.make_dot(y.mean(), params=dict(model.named_parameters()))

dot.render("model_diagram", format="png")


def generate_text(model, seed_text, num_tokens, max_length, temperature):
    model.eval()
    with torch.no_grad():
        # Initialize tokenizer instance
        tokenizer_instance = get_tokenizer('basic_english')

        # Tokenize the seed text
        tokens = tokenizer_instance(seed_text)

        # Convert tokens to numericalized form
        numericalized_tokens = [
            vocab[token] if token in vocab else vocab[UNK_TOKEN] for token in tokens]
        numericalized_tokens = torch.tensor(
            numericalized_tokens).unsqueeze(0).to(device)

        # Generate new tokens
        generated_tokens = []
        for _ in range(num_tokens):
            if len(generated_tokens) >= max_length:
                break

            # Generate output
            output = model(numericalized_tokens)

            # Get probabilities
            probabilities = nn.functional.softmax(
                output[0, -1] / temperature, dim=0)

            # Sample the next token
            next_token = torch.multinomial(probabilities, 1).item()

            # Append the new token to the list of tokens
            generated_tokens.append(next_token)

            # Prepare input for next iteration
            numericalized_tokens = torch.cat(
                [numericalized_tokens, torch.tensor([[next_token]]).to(device)], dim=1)

        # Truncate the generated tokens if it exceeds max_length
        generated_tokens = generated_tokens[:max_length]

        # Convert numericalized tokens back to text
        generated_text = ' '.join([vocab.get_itos()[token]
                                  for token in generated_tokens])

        # Limit the length of the generated text
        return generated_text[:max_length]


def interact_with_chatbot(model, tokenizer, prompt_prefix, max_seq_len=128, num_tokens=100, max_length=1024, temperature=8):
    print("Let's GO! (type 'quit' to exit)")
    while True:
        # Prompt user for input
        user_input = input("You: ")
        if user_input == "quit":
            break

        # Generate response from the model
        response = generate_text(
            model, user_input, num_tokens, max_length, temperature)

        # Decode the generated response and print it
        print("Chatbot:", response)


# Interaction with the chatbot
tokenizer_instance = get_tokenizer('basic_english')
interact_with_chatbot(model, tokenizer_instance, "User: ",
                      temperature=generation_params['temperature'])
