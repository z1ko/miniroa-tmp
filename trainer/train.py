from tqdm import tqdm
import torch
from trainer.train_builder import TRAINER
import einops as ein

from criterions.loss import calculate_multi_loss

@TRAINER.register("TAS")
def train_one_epoch(trainloader, model, criterion, optimizer, scaler, epoch, alpha, weights, writer=None, scheduler=None):

    loss_dict = {
        'verb': 0.0,
        'noun': 0.0
    }

    epoch_loss = 0
    batch_count = 0

    for it, (rgb_input, poses, target) in enumerate(tqdm(trainloader, desc=f'Epoch:{epoch} Training', postfix=f'lr: {optimizer.param_groups[0]["lr"]:.7f}')):
        rgb_input, poses, target = rgb_input.cuda(), poses.cuda(), target.cuda()
        batch_count += 1

        # NOTE: To remove
        #if batch_count == 1:
        #    break

        model.train()
        if scaler != None:
            with torch.cuda.amp.autocast():    
                out_dict = model(rgb_input, poses) 
                loss = criterion(out_dict, target)   
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:

            outputs = model(rgb_input, poses)
            target_verb, target_noun = torch.split(target, split_size_or_sections=1, dim=-1)
            target_verb = torch.squeeze(target_verb)
            target_noun = torch.squeeze(target_noun)
            target = [target_verb, target_noun]

            categories = [('verb', 25), ('noun', 91)]
            losses, combined_loss = calculate_multi_loss(outputs, target, categories, alpha, weights)
            loss_dict['verb'] += losses['verb']['loss_total']
            loss_dict['noun'] += losses['noun']['loss_total']

            # Print

            #logits = out_dict['logits']
            #logits = ein.rearrange(logits, "B T C -> B C T")
            #loss = criterion(logits, target)

            optimizer.zero_grad(set_to_none=True)
            #loss.backward()
            combined_loss.backward()
            optimizer.step()

        epoch_loss += combined_loss.item()
        if writer != None:
            writer.add_scalar("Train Loss", loss.item(), it+epoch*len(trainloader))

    epoch_loss /= batch_count
    loss_dict['verb'] /= batch_count
    loss_dict['noun'] /= batch_count

    return loss_dict, epoch_loss

@TRAINER.register("OAD")
def train_one_epoch(trainloader, model, criterion, optimizer, scaler, epoch, writer=None, scheduler=None):
    epoch_loss = 0
    for it, (rgb_input, flow_input, target) in enumerate(tqdm(trainloader, desc=f'Epoch:{epoch} Training', postfix=f'lr: {optimizer.param_groups[0]["lr"]:.7f}')):
        rgb_input, flow_input, target = rgb_input.cuda(), flow_input.cuda(), target.cuda()
        model.train()
        if scaler != None:
            with torch.cuda.amp.autocast():    
                out_dict = model(rgb_input, flow_input) 
                loss = criterion(out_dict, target)   
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out_dict = model(rgb_input, flow_input) 
            loss = criterion(out_dict, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        if writer != None:
            writer.add_scalar("Train Loss", loss.item(), it+epoch*len(trainloader))
    return epoch_loss

@TRAINER.register("ANTICIPATION")
def ant_train_one_epoch(trainloader, model, criterion, optimizer, scaler, epoch, writer=None, scheduler=None):
    epoch_loss = 0
    for it, (rgb_input, flow_input, target, ant_target) in enumerate(tqdm(trainloader, desc=f'Epoch:{epoch} Training', postfix=f'lr: {optimizer.param_groups[0]["lr"]:.7f}')):
        rgb_input, flow_input, target, ant_target = rgb_input.cuda(), flow_input.cuda(), target.cuda(), ant_target.cuda()
        model.train()
        if scaler != None:
            with torch.cuda.amp.autocast():    
                out_dict = model(rgb_input, flow_input) 
                loss = criterion(out_dict, target, ant_target)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else: 
            out_dict = model(rgb_input, flow_input) 
            loss = criterion(out_dict, target, ant_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        epoch_loss += loss.item()
        if writer != None:
            writer.add_scalar("Train Loss", loss.item(), it+epoch*len(trainloader))
    return epoch_loss