import numpy as np
import torch
from torch import nn
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset

"""
Code adapted from pytorch examples to custom training loop and optimizers
https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
"""

def prepare_dataset(split="train", batch_size=64, train_frac=0.95):
    tokenizer = get_tokenizer('basic_english')
    data = AG_NEWS(split=split)

    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    # Transform word => int index
    vocab = build_vocab_from_iterator(yield_tokens(data), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    def text_pipeline(x):
        return vocab(tokenizer(x))

    def label_pipeline(x):
        return int(x) - 1

    def collate_batch(batch):
        label_list = []
        text_list = []
        offsets = [0]
        for (_label, _text) in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(device), text_list.to(device), offsets.to(device)

    train_iter, test_iter = AG_NEWS()
    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)
    num_train = int(len(train_dataset) * train_frac)
    split_train_, split_valid_ = \
        random_split(train_dataset, [num_train, len(train_dataset) - num_train])

    train_dataloader = DataLoader(split_train_, batch_size=batch_size,
                                  shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(split_valid_, batch_size=batch_size,
                                  shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=True, collate_fn=collate_batch)

    return vocab, train_dataloader, valid_dataloader, test_dataloader


class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim=64, num_class=4):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


def training_step(
        classifier,
        optimizer,
        criterion,
        batch,
        device,
        gradient_norm_clip=0.1
):
    label, text, offsets = batch
    label = label.to(device)
    text = text.to(device)
    offsets = offsets.to(device)

    classifier.train()
    optimizer.zero_grad()

    predicted_label = classifier(text, offsets)
    loss = criterion(predicted_label, label)
    loss.backward()
    accuracy = (predicted_label.argmax(1) == label).sum()

    # For stability
    torch.nn.utils.clip_grad_norm_(classifier.parameters(), gradient_norm_clip)
    optimizer.step()

    return {
        "loss": loss.detach().cpu(),
        "accuracy": accuracy.detach().cpu()
    }


# Compute accuracy
def evaluation_step(
        classifier,
        dataloader,
        device,
):
    classifier.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for label, text, offsets in dataloader:
            label = label.to(device)
            text = text.to(device)
            offsets = offsets.to(device)

            predicted_label = classifier(text, offsets)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count


# Set hyperparameters
device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 64
embed_dim = 64
num_class = 4
epochs = 50
lr = 1
gradient_norm_clip = 0.1
log_every = 50
T = 4

vocab, train_loader, valid_loader, test_loader = prepare_dataset()

classifier = TextClassificationModel(len(vocab), embed_dim, num_class).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer_1 = torch.optim.SGD(classifier.parameters(), lr=lr)
optimizer_2 = torch.optim.Adam(classifier.parameters(), lr=lr)

# Scheduler to decrease learning rate when performance plateaus
scheduler_1 = torch.optim.lr_scheduler.StepLR(optimizer_1, 10, gamma=0.1)
scheduler_2 = torch.optim.lr_scheduler.StepLR(optimizer_2, 10, gamma=0.1)

train_loss = []
train_acc = []
valid_acc = []

# Training loop
print("Training started")
for e in range(epochs):
    if e < T:
        optimizer = optimizer_1
    else:
        optimizer = optimizer_2

    for idx, batch in enumerate(train_loader):
        logs = training_step(
            classifier,
            optimizer,
            criterion,
            batch,
            device,
            gradient_norm_clip
        )
        if idx % log_every == 0:
            print(
                f"[{e}/{epochs}]"
                f"[{idx}/{len(train_loader)}]"
                f"\tAccuracy: {logs['accuracy'] :.4f}"
                f"\tLoss: {logs['loss'] :.4f}")
        train_loss.append(logs["loss"])
        train_acc.append(logs["accuracy"])

    val_epoch = evaluation_step(
        classifier,
        valid_loader,
        device
    )

    valid_acc.append(val_epoch)
    print(f"Validation step [{e}/{epochs}]"
          f"\tAccuracy: {val_epoch :.4f}")
    print("-----------------------------")
    scheduler_1.step()
    scheduler_2.step()


final_acc = evaluation_step(
    classifier,
    test_loader,
    device
)

print(f"Final accuracy: {final_acc:.4f}")
np.save("asdsad.npy", train_loss)
np.save("asdasd.npy", train_acc)
np.save("asdafd.npy", valid_acc)
