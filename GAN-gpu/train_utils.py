import numpy as np
import torch.optim as optimizer
from torch import nn
import torch
from generate_rap import generate_rap
from sentence_transformers import SentenceTransformer

sen_embed = SentenceTransformer('bert-base-nli-mean-tokens')

def calculate_joint_loss(discriminator_res):
    discriminator_res = discriminator_res.detach().cpu().numpy()
    g_loss = np.log(abs(discriminator_res))
    d_loss = np.log(abs(1 - discriminator_res))
    return torch.tensor(g_loss).cuda(), torch.tensor(d_loss).cuda()


def get_joint_performance(generator, discriminator, val_loader, device, max_words, final_dataset):
    val_g_loss = 0
    val_d_loss = 0
    y_pred = []
    y_true = []
    for (X, y) in val_loader:
        sen_input = final_dataset.get_sen(X)
        num_sentences = 1
        generated_lyrics = generate_rap(generator, sen_input, num_sentences, max_words, final_dataset)
        discriminator_res = discriminator(sen_embed.encode(generated_lyrics))

        g_loss, d_loss = calculate_joint_loss(discriminator_res)
        val_g_loss += g_loss
        val_d_loss += d_loss
        y_true.append(0)
        if discriminator_res > 0:
            y_pred.append(1)
        else:
            y_pred.append(0)

    val_d_acc  = (np.array(y_pred) * np.array(y_true)).sum() / len(y_pred)
    return val_d_acc, val_g_loss.item(), val_d_loss.item()


def get_discriminator_performance(model, loss_fn, val_loader, device):
    model.eval()
    y_true = []  # true labels
    y_pred = []  # predicted labels
    total_loss = []  # loss for each batch

    with torch.no_grad():
        for (X, y) in val_loader:
            res = model(X.to(device='cuda')).to(device='cuda')
            loss = loss_fn(res, y.clone().detach().float().to(device='cuda'))
            res[res > 0] = 1
            res[res < 0] = 0

            total_loss.append(loss.item())
            y_true.append(y)
            y_pred.append(res.clone().detach())

    y_true = torch.ravel(torch.cat(y_true))
    y_pred = torch.ravel(torch.cat(y_pred))
    accuracy = (y_true == y_pred).sum() / y_pred.shape[0]
    total_loss = sum(total_loss) / len(total_loss)

    return accuracy, total_loss


def get_generator_loss(model, loss_fn, val_loader, state):
    model.eval()
    total_loss = []  # loss for each batch
    with torch.no_grad():
        for (X, y) in val_loader:
            res, (state_h, state_c)  = model(X, state)
            loss = loss_fn(res.transpose(1, 2), y.clone().detach())
            total_loss.append(loss.item())

    total_loss = sum(total_loss) / len(total_loss)

    return total_loss


def get_optimizer(net, net_type, optim_type, lr, weight_decay):
    if net_type == 'generator':
        if optim_type == 'adam':
            return optimizer.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    if net_type == 'discriminator':
        if optim_type == 'adam':
            return optimizer.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)


def get_loss_fn(net_type, loss_type):
    if net_type == 'generator':
        if loss_type == 'bce':
            return nn.BCEWithLogitsLoss()
        if loss_type == 'cross':
            return nn.CrossEntropyLoss()
    if net_type == 'discriminator':
        if loss_type == 'bce':
            return nn.BCEWithLogitsLoss()
        if loss_type == 'cross':
            return nn.CrossEntropyLoss()
    return nn.BCEWithLogitsLoss()