from tarfile import XGLTYPE
import torch
import time
import copy
from train_utils import get_optimizer, get_loss_fn, get_generator_loss, get_discriminator_performance, calculate_joint_loss, get_joint_performance
from generate_rap import generate_rap
from sentence_transformers import SentenceTransformer

sen_embed = SentenceTransformer('bert-base-nli-mean-tokens')

def pre_train_discriminator(args, model, dataloader, val_loader, loss_type, optim_type, lr, weight_decay, patience, device):
    model.train()

    optimizer = get_optimizer(model, 'discriminator', optim_type, lr, weight_decay)
    loss_fn = get_loss_fn('discriminator', loss_type)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0, total_iters=args["pre_train_epochs"])

    best_model, best_accuracy = None, 0
    num_bad_epoch = 0
    for epoch in range(args["pre_train_epochs"]):
        for batch, (X, y) in enumerate(dataloader):
            y_pre = model(X)
            loss = loss_fn(y_pre, torch.Tensor(y))
            optimizer.zero_grad()
            loss.requires_grad_().backward()
            optimizer.step()

            print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })

        accuracy, _ = get_discriminator_performance(model, loss_fn, val_loader, device)
        if accuracy > best_accuracy:
            best_model = copy.deepcopy(model)
            best_accuracy = accuracy
            num_bad_epoch = 0
        else:
            num_bad_epoch += 1

        # early stopping
        if num_bad_epoch >= patience:
            break

        # learning rate scheduler
        scheduler.step()

    return best_model


def pre_train_generator(args, model, dataloader, val_loader, loss_type, optim_type, lr, weight_decay, patience, device):
    model.train()

    optimizer = get_optimizer(model, 'generator', optim_type, lr, weight_decay)
    loss_fn = get_loss_fn('generator', loss_type)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0, total_iters=args["pre_train_epochs"])

    best_model, best_loss = None, 100
    num_bad_epoch = 0
    for epoch in range(args["pre_train_epochs"]):
        state_h, state_c = model.init_state(args["sequence_length"])

        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()

            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = loss_fn(y_pred.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.requires_grad_().backward()
            optimizer.step()

            print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })

        loss = get_generator_loss(model, loss_fn, val_loader, (state_h, state_c))
        if loss < best_loss:
            best_model = copy.deepcopy(model)
            best_loss = loss
            num_bad_epoch = 0
        else:
            num_bad_epoch += 1

        # early stopping
        if num_bad_epoch >= patience:
            break

        # learning rate scheduler
        scheduler.step()

    return best_model


def train_model(generator, discriminator, input_loader, input_data, num_epoch, g_para, d_para, val_loader, patience, max_words, device):
    generator.train()
    discriminator.train()

    dis_loss_all, gen_loss_all, loss_ind = [], [], []
    num_itr = 0
    best_gen, best_dis, best_dis_loss, best_gen_loss = generator, discriminator, float('-inf'), float('inf')
    num_bad_epoch = 0

    g_optim = get_optimizer(generator, 'generator', g_para["optim_type"], g_para["lr"], g_para["weight_decay"])
    d_optim = get_optimizer(discriminator, 'discriminator', d_para["optim_type"], d_para["lr"], d_para["weight_decay"])
    # g_loss_fn = get_loss_fn('generator', g_para["loss_type"])
    # d_loss_fn = get_loss_fn('discriminator', d_para["loss_type"])

    g_scheduler = torch.optim.lr_scheduler.LinearLR(g_optim, start_factor=1.0, end_factor=0, total_iters=num_epoch)
    d_scheduler = torch.optim.lr_scheduler.LinearLR(d_optim, start_factor=1.0, end_factor=0, total_iters=num_epoch)

    print('------------------------ Start Training ------------------------')
    t_start = time.time()
    g_loss, d_loss = 0, 0
    for epoch in range(num_epoch):
        for batch, (X, y) in enumerate(input_loader):
            print(batch)
            num_itr += 1

            sen_input = input_data.get_sen(X)
            num_sentences = 1
            generated_lyrics = generate_rap(generator, sen_input, num_sentences, max_words, input_data)[0]

            discriminator_res = discriminator(sen_embed.encode(generated_lyrics))

            g_optim.zero_grad()
            g_loss, d_loss = calculate_joint_loss(copy.copy(discriminator_res))
            g_loss.requires_grad_().backward()
            g_optim.step()

            d_optim.zero_grad()
            d_loss.requires_grad_().backward()
            d_optim.step()

            dis_loss_all.append(d_loss.item())
            gen_loss_all.append(g_loss.item())
            loss_ind.append(num_itr)

        print('Epoch No. {0} -- generator loss = {1:.4f} -- discriminator loss = {2:.4f}'.format(
            epoch + 1,
            g_loss.item(),
            d_loss.item()
        ))

        # Validation:
        val_d_acc, val_g_loss, val_d_loss = get_joint_performance(generator, discriminator, val_loader, device, max_words, input_data)
        print("Validation Generator loss: {:.4f}".format(val_g_loss))
        print("Validation Discriminator loss: {:.4f}".format(val_d_loss))

        if val_d_loss > best_dis_loss and val_g_loss < best_gen_loss:
            best_dis = copy.deepcopy(discriminator)
            best_gen = copy.deepcopy(generator)
            best_dis_loss = d_loss
            best_gen_loss = g_loss
            num_bad_epoch = 0
        else:
            num_bad_epoch += 1

        # early stopping
        if num_bad_epoch >= patience:
            break
        if val_d_acc < 0.5:
            break

        # learning rate scheduler
        g_scheduler.step()
        d_scheduler.step()

    t_end = time.time()
    print('Training lasted {0:.2f} minutes'.format((t_end - t_start) / 60))
    print('------------------------ Training Done ------------------------')
    stats = {'dis_loss': dis_loss_all,
             'gen_loss': gen_loss_all,
             'loss_ind': loss_ind,
             }

    return best_gen, best_dis, stats
