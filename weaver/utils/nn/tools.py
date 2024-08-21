import numpy as np
import awkward as ak
import tqdm
import time
import torch

from collections import defaultdict, Counter
from .metrics import evaluate_metrics
from ..data.tools import _concat
from ..logger import _logger


def _flatten_label(label, mask=None):
    if label.ndim > 1:
        label = label.view(-1)
        if mask is not None:
            label = label[mask.view(-1)]
    # print('label', label.shape, label)
    return label


def _flatten_preds(model_output, label=None, mask=None, label_axis=1):
    if not isinstance(model_output, tuple):
        # `label` and `mask` are provided as function arguments
        preds = model_output
    else:
        if len(model_output == 2):
            # use `mask` from model_output instead
            # `label` still provided as function argument
            preds, mask = model_output
        elif len(model_output == 3):
            # use `label` and `mask` from model output
            preds, label, mask = model_output

    # preds: (N, num_classes); (N, num_classes, P)
    # label: (N,);             (N, P)
    # mask:  None;             (N, P) / (N, 1, P)
    if preds.ndim > 2:
        preds = preds.transpose(label_axis, -1).contiguous()
        preds = preds.view((-1, preds.shape[-1]))
        if mask is not None:
            preds = preds[mask.view(-1)]
    # print('preds', preds.shape, preds)

    if label is not None:
        label = _flatten_label(label, mask)

    return preds, label, mask

def train_classification(
        model, loss_func, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=None, grad_scaler=None,
        tb_helper=None):
    model.train()

    data_config = train_loader.dataset.config

    label_counter = Counter()
    total_loss = 0
    num_batches = 0
    total_correct = 0
    entry_count = 0
    count = 0
    start_time = time.time()
    with tqdm.tqdm(train_loader) as tq:
        for X, y_cat, _, _ in tq:
            inputs = [X[k].to(dev) for k in data_config.input_names]
            label = y_cat[data_config.label_names[0]].long().to(dev)
            entry_count += label.shape[0]
            try:
                mask = y_cat[data_config.label_names[0] + '_mask'].bool().to(dev)
            except KeyError:
                mask = None
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                model_output = model(*inputs)
                logits, label, _ = _flatten_preds(model_output, label=label, mask=mask)
                loss = loss_func(logits, label)
            if grad_scaler is None:
                loss.backward()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()

            if scheduler and getattr(scheduler, '_update_per_step', False):
                scheduler.step()

            loss  = loss.detach().item()
            label = label.detach();
            _, preds = logits.detach().max(1)

            num_examples = label.shape[0]
            label_counter.update(label.numpy(force=True))
            num_batches += 1
            count += num_examples
            correct = (preds == label).sum().item()
            total_loss += loss
            total_correct += correct

            tq.set_postfix({
                'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                'Loss': '%.5f' % loss,
                'AvgLoss': '%.5f' % (total_loss / num_batches),
                'Acc': '%.5f' % (correct / num_examples),
                'AvgAcc': '%.5f' % (total_correct / count)})

            if tb_helper:
                tb_helper.write_scalars([
                    ("Loss/train", loss, tb_helper.batch_train_count + num_batches),
                    ("Acc/train", correct / num_examples, tb_helper.batch_train_count + num_batches),
                ])
                if tb_helper.custom_fn:
                    with torch.no_grad():
                        tb_helper.custom_fn(model_output=model_output, model=model,
                                            epoch=epoch, i_batch=num_batches, mode='train')

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (entry_count, entry_count / time_diff))
    _logger.info('Train AvgLoss: %.5f, AvgAcc: %.5f' % (total_loss / num_batches, total_correct / count))
    _logger.info('Train class distribution: \n    %s', str(sorted(label_counter.items())))

    if tb_helper:
        tb_helper.write_scalars([
            ("Loss/train (epoch)", total_loss / num_batches, epoch),
            ("Acc/train (epoch)", total_correct / count, epoch),
        ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode='train')
        # update the batch state
        tb_helper.batch_train_count += num_batches

    if scheduler and not getattr(scheduler, '_update_per_step', False):
        scheduler.step()


def evaluate_classification(model, test_loader, dev, epoch, for_training=True, loss_func=None, steps_per_epoch=None,
                            eval_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix'],
                            tb_helper=None):
    model.eval()

    data_config = test_loader.dataset.config

    label_counter = Counter()
    total_loss = 0
    num_batches = 0
    total_correct = 0
    entry_count = 0
    count = 0
    scores = []
    labels_counts = []
    labels = defaultdict(list)
    targets = defaultdict(list)
    observers = defaultdict(list)
    start_time = time.time()
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y_cat, _, Z in tq:
                # X, y_cat, _: torch.Tensor; Z: ak.Array
                inputs = [X[k].to(dev) for k in data_config.input_names]
                label = y_cat, _[data_config.label_names[0]].long().to(dev)
                entry_count += label.shape[0]
                try:
                    mask = y_cat, _[data_config.label_names[0] + '_mask'].bool().to(dev)
                except KeyError:
                    mask = None
                model_output = model(*inputs)
                logits, label, mask = _flatten_preds(model_output, label=label, mask=mask)
                scores.append(torch.softmax(logits.float(), dim=1).numpy(force=True))

                if mask is not None:
                    mask = mask.cpu()
                for k, v in y_cat.items():
                    labels[k].append(_flatten_label(v, mask).numpy(force=True))
                if not for_training:
                    for k, v in Z.items():
                        observers[k].append(v)

                num_examples = label.shape[0]
                label_counter.update(label.numpy(force=True))
                if not for_training and mask is not None:
                    labels_counts.append(np.squeeze(mask.numpy(force=True).sum(axis=-1)))

                _, preds = logits.max(1)
                loss = 0 if loss_func is None else loss_func(logits, label).item()

                num_batches += 1
                count += num_examples
                correct = (preds == label).sum().item()
                total_loss += loss * num_examples
                total_correct += correct

                tq.set_postfix({
                    'Loss': '%.5f' % loss,
                    'AvgLoss': '%.5f' % (total_loss / count),
                    'Acc': '%.5f' % (correct / num_examples),
                    'AvgAcc': '%.5f' % (total_correct / count)})

                if tb_helper:
                    if tb_helper.custom_fn:
                        with torch.no_grad():
                            tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch,
                                                i_batch=num_batches, mode='eval' if for_training else 'test')

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (entry_count, entry_count / time_diff))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_counter.items())))

    if tb_helper:
        tb_mode = 'eval' if for_training else 'test'
        tb_helper.write_scalars([
            ("Loss/%s (epoch)" % tb_mode, total_loss / count, epoch),
            ("Acc/%s (epoch)" % tb_mode, total_correct / count, epoch),
        ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode=tb_mode)

    scores = np.concatenate(scores)
    labels = {k: _concat(v) for k, v in labels.items()}
    metric_results = evaluate_metrics(labels[data_config.label_names[0]], scores, eval_metrics=eval_metrics)
    _logger.info('Evaluation metrics: \n%s', '\n'.join(
        ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))

    if for_training:
        return total_correct / count
    else:
        # convert 2D labels/scores
        if len(scores) != entry_count:
            if len(labels_counts):
                labels_counts = np.concatenate(labels_counts)
                scores = ak.unflatten(scores, labels_counts)
                for k, v in labels.items():
                    labels[k] = ak.unflatten(v, labels_counts)
            else:
                assert (count % entry_count == 0)
                scores = scores.reshape((entry_count, int(count / entry_count), -1)).transpose((1, 2))
                for k, v in labels.items():
                    labels[k] = v.reshape((entry_count, -1))
        observers = {k: _concat(v) for k, v in observers.items()}
        return total_correct / count, scores, labels, targets, observers


def evaluate_onnx(model_path, test_loader, eval_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix']):
    import onnxruntime
    sess = onnxruntime.InferenceSession(model_path)

    data_config = test_loader.dataset.config

    label_counter = Counter()
    total_correct = 0
    count = 0
    scores = []
    labels = defaultdict(list)
    targets = defaultdict(list)
    observers = defaultdict(list)
    start_time = time.time()
    with tqdm.tqdm(test_loader) as tq:
        for X, y_cat, _, Z in tq:
            # X, y_cat: torch.Tensor; Z: ak.Array
            inputs = {k: v.numpy(force=True) for k, v in X.items()}
            label = y_cat[data_config.label_names[0]].numpy(force=True)
            num_examples = label.shape[0]
            label_counter.update(label)
            score = sess.run([], inputs)[0]
            preds = score.argmax(1)

            scores.append(score)
            for k, v in y_cat.items():
                labels[k].append(v.numpy(force=True))
            for k, v in Z.items():
                observers[k].append(v)

            correct = (preds == label).sum()
            total_correct += correct
            count += num_examples

            tq.set_postfix({
                'Acc': '%.5f' % (correct / num_examples),
                'AvgAcc': '%.5f' % (total_correct / count)})

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_counter.items())))

    scores = np.concatenate(scores)
    labels = {k: _concat(v) for k, v in labels.items()}
    metric_results = evaluate_metrics(labels[data_config.label_names[0]], scores, eval_metrics=eval_metrics)
    _logger.info('Evaluation metrics: \n%s', '\n'.join(
        ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))
    observers = {k: _concat(v) for k, v in observers.items()}
    return total_correct / count, scores, labels, targets, observers


def train_regression(
        model, loss_func, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=None, grad_scaler=None,
        tb_helper=None):
    model.train()

    data_config = train_loader.dataset.config

    total_loss = 0
    num_batches = 0
    sum_abs_err = 0
    sum_sqr_err = 0
    count = 0
    start_time = time.time()
    with tqdm.tqdm(train_loader) as tq:
        for X, _, y_reg, _ in tq:
            inputs = [X[k].to(dev) for k in data_config.input_names]
            for idx, names in enumerate(data_config.target_names):
                if idx == 0:
                    target = y_reg[names].float();
                else:
                    target = torch.column_stack((target,y_reg[names].float()))
            num_examples = target.shape[0]
            target = target.to(dev)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                model_output = model(*inputs)
                preds = model_output.squeeze()
                loss = loss_func(preds, target)
            if grad_scaler is None:
                loss.backward()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()

            if scheduler and getattr(scheduler, '_update_per_step', False):
                scheduler.step()

            loss = loss.item()

            num_batches += 1
            count += num_examples
            total_loss += loss
            e = preds - target
            abs_err = e.abs().sum().item()
            sum_abs_err += abs_err
            sqr_err = e.square().sum().item()
            sum_sqr_err += sqr_err

            tq.set_postfix({
                'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                'Loss': '%.5f' % loss,
                'AvgLoss': '%.5f' % (total_loss / num_batches),
                'MSE': '%.5f' % (sqr_err / num_examples),
                'AvgMSE': '%.5f' % (sum_sqr_err / count),
                'MAE': '%.5f' % (abs_err / num_examples),
                'AvgMAE': '%.5f' % (sum_abs_err / count),
            })

            if tb_helper:
                tb_helper.write_scalars([
                    ("Loss/train", loss, tb_helper.batch_train_count + num_batches),
                    ("MSE/train", sqr_err / num_examples, tb_helper.batch_train_count + num_batches),
                    ("MAE/train", abs_err / num_examples, tb_helper.batch_train_count + num_batches),
                ])
                if tb_helper.custom_fn:
                    with torch.no_grad():
                        tb_helper.custom_fn(model_output=model_output, model=model,
                                            epoch=epoch, i_batch=num_batches, mode='train')

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info('Train AvgLoss: %.5f, AvgMSE: %.5f, AvgMAE: %.5f' %
                 (total_loss / num_batches, sum_sqr_err / count, sum_abs_err / count))

    if tb_helper:
        tb_helper.write_scalars([
            ("Loss/train (epoch)", total_loss / num_batches, epoch),
            ("MSE/train (epoch)", sum_sqr_err / count, epoch),
            ("MAE/train (epoch)", sum_abs_err / count, epoch),
        ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode='train')
        # update the batch state
        tb_helper.batch_train_count += num_batches

    if scheduler and not getattr(scheduler, '_update_per_step', False):
        scheduler.step()


def evaluate_regression(model, test_loader, dev, epoch, for_training=True, loss_func=None, steps_per_epoch=None,
                        eval_metrics=['mean_squared_error', 'mean_absolute_error', 'median_absolute_error',
                                      'mean_gamma_deviance'],
                        tb_helper=None):
    model.eval()

    data_config = test_loader.dataset.config

    total_loss = 0
    num_batches = 0
    sum_sqr_err = 0
    sum_abs_err = 0
    count = 0
    scores = []
    labels = defaultdict(list)
    targets = defaultdict(list)
    observers = defaultdict(list)
    start_time = time.time()
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, _, y_reg, Z in tq:
                # X, y_reg: torch.Tensor; Z: ak.Array
                inputs = [X[k].to(dev) for k in data_config.input_names]
                for idx, names in enumerate(data_config.target_names):
                    if idx == 0:
                        target = y_reg[names].float();
                    else:
                        target = torch.column_stack((target,y_reg[names].float()))
                num_examples = target.shape[0]
                target = target.to(dev)
                model_output = model(*inputs)
                preds = model_output.squeeze().float()

                scores.append(preds.numpy(force=True))
                for k, v in y_reg.items():
                    targets[k].append(v.numpy(force=True))
                if not for_training:
                    for k, v in Z.items():
                        observers[k].append(v)

                loss = 0 if loss_func is None else loss_func(preds, target).item()

                num_batches += 1
                count += num_examples
                total_loss += loss
                e = preds - target
                abs_err = e.abs().sum().item()
                sum_abs_err += abs_err
                sqr_err = e.square().sum().item()
                sum_sqr_err += sqr_err

                tq.set_postfix({
                    'Loss': '%.5f' % loss,
                    'AvgLoss': '%.5f' % (total_loss / count),
                    'MSE': '%.5f' % (sqr_err / num_examples),
                    'AvgMSE': '%.5f' % (sum_sqr_err / count),
                    'MAE': '%.5f' % (abs_err / num_examples),
                    'AvgMAE': '%.5f' % (sum_abs_err / count),
                })

                if tb_helper:
                    if tb_helper.custom_fn:
                        with torch.no_grad():
                            tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch,
                                                i_batch=num_batches, mode='eval' if for_training else 'test')

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))

    if tb_helper:
        tb_mode = 'eval' if for_training else 'test'
        tb_helper.write_scalars([
            ("Loss/%s (epoch)" % tb_mode, total_loss / count, epoch),
            ("MSE/%s (epoch)" % tb_mode, sum_sqr_err / count, epoch),
            ("MAE/%s (epoch)" % tb_mode, sum_abs_err / count, epoch),
        ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode=tb_mode)

    scores = np.concatenate(scores)
    targets = {k: _concat(v) for k, v in targets.items()}

    for idx, (name,element) in enumerate(targets.items()):
        if len(data_config.target_names) == 1:
            metric_results = evaluate_metrics(element, scores, eval_metrics=eval_metrics)
        else:
            metric_results = evaluate_metrics(element, scores[:,idx], eval_metrics=eval_metrics)

        _logger.info('Evaluation metrics: \n%s', '\n'.join(
            ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))

    if for_training:
        return total_loss / count
    else:
        # convert 2D labels/scores
        observers = {k: _concat(v) for k, v in observers.items()}
        # scores = scores.reshape(len(scores),len(data_config.target_names))
        return total_loss / count, scores, labels, targets, observers

def train_classreg(model, loss_func, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=None, grad_scaler=None, tb_helper=None, network_option=None):
    
    model.train()
    torch.backends.cudnn.benchmark = True;
    torch.backends.cudnn.enabled = True;
    
    data_config = train_loader.dataset.config
   
    num_batches, total_loss, total_cat_loss, total_reg_loss, count = 0, 0, 0, 0, 0
    label_counter = Counter()
    total_correct, sum_sqr_err = 0, 0
    inputs, target, label, model_output, label_mask = None, None, None, None, None;
    loss, loss_cat, loss_reg, pred_cat, pred_reg, residual_reg, correct = None, None, None, None, None, None, None;
    loss_contrastive, model_output_contrastive, total_contrastive_loss = None, None, 0;

    num_labels  = len(data_config.label_value);
    if type(data_config.target_value) == dict:
        num_targets = sum(len(dct) if type(dct) == list else 1 for dct in data_config.target_value.values())
    else:
        num_targets = len(data_config.target_value);

    network_options = None;
    if network_option:
        network_options = {k: ast.literal_eval(v) for k, v in network_option}

    start_time = time.time()

    with tqdm.tqdm(train_loader) as tq:
        for X, y_cat, y_reg, _ in tq:
            ### input features for the model
            inputs = [X[k].to(dev,non_blocking=True) for k in data_config.input_names]
            ### build classification true labels (numpy argmax)
            label = y_cat[data_config.label_names[0]].long().to(dev,non_blocking=True)
            try:
                label_mask = y_cat[data_config.label_names[0] + '_mask'].bool().to(dev,non_blocking=True)
            except KeyError:
                label_mask = None;
            label = _flatten_label(label,mask=label_mask)
            label_counter.update(label.numpy(force=True).astype(dtype=np.int32))
            ### build regression targets
            for idx, names in enumerate(data_config.target_names):
                if idx == 0:
                    target = y_reg[names].float();
                else:
                    target = torch.column_stack((target,y_reg[names].float()))
            target = target.to(dev,non_blocking=True)
            ### Number of samples in the batch
            num_examples = max(label.shape[0],target.shape[0]);
            ### loss minimization
            model.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):            
                label = label.squeeze();
                target = target.squeeze();
                ### evaluate the model
                if network_options and network_options.get('use_contrastive',False):
                    model_output, model_output_contrastive = model(*inputs)         
                    model_output_contrastive = model_output_contrastive.squeeze().float();
                else:
                    model_output = model(*inputs)
                model_output_cat = model_output[:,:num_labels];
                model_output_reg = model_output[:,num_labels:num_labels+num_targets];
                model_output_cat, label, label_mask = _flatten_preds(model_output_cat,label=label,mask=label_mask)
                model_output_cat = model_output_cat.squeeze().float();
                model_output_reg = model_output_reg.squeeze().float();
            
            ### evaluate loss function
            if network_options and network_options.get('use_contrastive',False):                
                loss, loss_cat, loss_reg, loss_contrastive = loss_func(model_output_cat,label,model_output_reg,target,model_output_contrastive);
            else:
                loss, loss_cat, loss_reg = loss_func(model_output_cat,label,model_output_reg,target);

            if grad_scaler is None:
                loss.backward()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()

            if scheduler and getattr(scheduler, '_update_per_step', True):
                scheduler.step()

            ### evaluate loss function and counters
            loss = loss.detach().item()
            loss_cat = loss_cat.detach().item()
            loss_reg = loss_reg.detach().item()
            total_loss += loss
            total_cat_loss += loss_cat;
            total_reg_loss += loss_reg;
            if loss_contrastive:
                loss_contrastive = loss_contrastive.detach().item()
                total_contrastive_loss += loss_contrastive;
            num_batches += 1
            count += num_examples;
            
            ## take the classification prediction and compare with the true labels            
            label = label.detach()
            target = target.detach()
            _, pred_cat = model_output_cat.detach().max(1)
            correct  = (pred_cat == label).sum().item()
            total_correct += correct

            ## take the regression prediction and compare with true targets
            pred_reg = model_output_reg.detach().float();
            residual_reg = pred_reg - target;            
            sqr_err = residual_reg.square().sum().item()
            sum_sqr_err += sqr_err
            
            ### monitor metrics
            if network_options and network_options.get('use_contrastive',False):
                postfix = {
                    'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                    'Loss': '%.3f' % (total_loss / num_batches if num_batches else 0),
                    'LossCat': '%.3f' % (total_cat_loss / num_batches if num_batches else 0),
                    'LossReg': '%.3f' % (total_reg_loss / num_batches if num_batches else 0),
                    'LossCont': '%.3f' % (total_contrastive_loss / num_batches if num_batches else 0),
                    'AvgAccCat': '%.3f' % (total_correct / count if count else 0),
                    'AvgMSE': '%.3f' % (sum_sqr_err / count if count else 0),
                }
            else:
                postfix = {
                    'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                    'AvgLoss': '%.3f' % (total_loss / num_batches if num_batches else 0),
                    'AvgLossCat': '%.3f' % (total_cat_loss / num_batches if num_batches else 0), 
                    'AvgLossReg': '%.3f' % (total_reg_loss / num_batches if num_batches else 0),
                    'AvgAccCat': '%.3f' % (total_correct / count if count else 0),
                    'AvgMSE': '%.3f' % (sum_sqr_err / count if count else 0)
                }
            tq.set_postfix(postfix);

            if tb_helper:
                tb_helper.write_scalars([
                    ("Loss/train", loss, tb_helper.batch_train_count + num_batches),
                    ("Acc/train", correct / num_examples, tb_helper.batch_train_count + num_batches),
                    ("MSE/train", sqr_err / num_examples, tb_helper.batch_train_count + num_batches),
                ])
                if tb_helper.custom_fn:
                    with torch.no_grad():
                        tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches, mode='train')

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    ### training summary
    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info('Train AvgLoss: %.5f'% (total_loss / num_batches))
    _logger.info('Train AvgLoss Cat: %.5f'% (total_cat_loss / num_batches))
    _logger.info('Train AvgLoss Reg: %.5f'% (total_reg_loss / num_batches))
    if network_options and network_options.get('use_contrastive',False):
        _logger.info('Train AvgLoss Contrastive: %.5f'%(total_contrastive_loss / num_batches if num_batches else 0))
    _logger.info('Train AvgAcc: %.5f'%(total_correct / count))
    _logger.info('Train AvgMSE: %.5f'%(sum_sqr_err / count))
    _logger.info('Train class distribution: \n %s', str(sorted(label_counter.items())))

    if tb_helper:
        tb_helper.write_scalars([
            ("Loss/train (epoch)", total_loss / num_batches, epoch),
            ("Loss Cat/train (epoch)", total_cat_loss / num_batches, epoch),
            ("Loss Reg/train (epoch)", total_reg_loss / num_batches, epoch),
            ("Acc/train (epoch)", total_correct / count, epoch),
            ("MSE/train (epoch)", sum_sqr_err / count, epoch),
        ])
        if tb_helper.custom_fn:
         with torch.no_grad():
            tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode='train')
        # update the batch state
        tb_helper.batch_train_count += num_batches

def evaluate_classreg(model, test_loader, dev, epoch, for_training=True, loss_func=None, 
                      steps_per_epoch=None, tb_helper=None, network_option=None, grad_scaler=None,
                      eval_cat_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix'],
                      eval_reg_metrics=['mean_squared_error', 'mean_absolute_error', 'median_absolute_error', 'mean_gamma_deviance']):
    
    model.eval()
    torch.backends.cudnn.benchmark = True;
    torch.backends.cudnn.enabled = True;

    data_config = test_loader.dataset.config
    label_counter = Counter()
    total_loss, total_cat_loss, total_reg_loss, num_batches, total_correct, sum_sqr_err, entry_count, count = 0, 0, 0, 0, 0, 0, 0, 0;
    inputs, label, target,  model_output, pred_cat_output, pred_reg, loss, loss_cat, loss_reg, label_mask = None, None, None, None, None , None, None, None, None, None;
    inputs_grad_sign, network_options = None, None;
    scores_cat, scores_reg = [], [];
    labels, targets, observers = defaultdict(list), defaultdict(list), defaultdict(list);

    num_labels  = len(data_config.label_value);
    if type(data_config.target_value) == dict:
        num_targets = sum(len(dct) if type(dct) == list else 1 for dct in data_config.target_value.values())
    else:
        num_targets = len(data_config.target_value);

    if network_option:
        network_options = {k: ast.literal_eval(v) for k, v in network_option}
        
    start_time = time.time()

    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y_cat, y_reg, Z in tq:
                ### input features for the model
                inputs = [X[k].to(dev,non_blocking=True) for k in data_config.input_names]
                ### build classification true labels
                label  = y_cat[data_config.label_names[0]].long().to(dev,non_blocking=True)
                try:
                    label_mask = y_cat[data_config.label_names[0] + '_mask'].bool().to(dev,non_blocking=True)
                except KeyError:
                    label_mask = None
                label  = _flatten_label(label,mask=label_mask)
                label_counter.update(label.numpy(force=True).astype(dtype=np.int32))
                ### build regression targets
                for idx, names in enumerate(data_config.target_names):
                    if idx == 0:
                        target = y_reg[names].float();
                    else:
                        target = torch.column_stack((target,y_reg[names].float()))
                target = target.to(dev,non_blocking=True)
                ### update counters
                num_examples = max(label.shape[0],target.shape[0]);
                entry_count += num_examples

                ### define truth labels for classification and regression
                for k, name in enumerate(data_config.label_names):                    
                    labels[name].append(_flatten_label(y_cat[name],None).numpy(force=True).astype(dtype=np.int32))
                for k, name in enumerate(data_config.target_names):
                    targets[name].append(y_reg[name].numpy(force=True).astype(dtype=np.float32))                
                ### observers
                if not for_training:
                    for k, v in Z.items():
                        if v.numpy(force=True).dtype in (np.int16, np.int32, np.int64):
                            observers[k].append(v.numpy(force=True).astype(dtype=np.int32))
                        else:
                            observers[k].append(v.numpy(force=True).astype(dtype=np.float32))

                model_output = model(*inputs)

                ### build classification and regression outputs
                label  = label.squeeze();
                target = target.squeeze();
                model_output_cat = model_output[:,:num_labels];
                model_output_reg = model_output[:,num_labels:num_labels+num_targets];
                model_output_cat, label, label_mask = _flatten_preds(model_output_cat,label=label,mask=label_mask)
                model_output_cat = model_output_cat.squeeze().float();
                model_output_reg = model_output_reg.squeeze().float();

                ### save scores
                if model_output_cat.shape[0] == num_examples and model_output_reg.shape[0] == num_examples:
                    scores_cat.append(torch.softmax(model_output_cat,dim=1).numpy(force=True).astype(dtype=np.float32));
                    scores_reg.append(model_output_reg.numpy(force=True).astype(dtype=np.float32))
                else:
                    scores_cat.append(torch.zeros(num_examples,num_labels).numpy(force=True).astype(dtype=np.float32));
                    if num_targets > 1:
                        scores_reg.append(torch.zeros(num_examples,num_targets).numpy(force=True).astype(dtype=np.float32));
                    else:
                        scores_reg.append(torch.zeros(num_examples).numpy(force=True).astype(dtype=np.float32));                        
                ### evaluate loss function
                if loss_func != None:
                    ### true labels and true target 
                    if network_options and network_options.get('use_contrastive',False):
                        loss, loss_cat, loss_reg, _ = loss_func(model_output_cat,label,model_output_reg,target)
                    else:
                        loss, loss_cat, loss_reg = loss_func(model_output_cat,label,model_output_reg,target)

                    loss = loss.item()
                    loss_cat = loss_cat.item()
                    loss_reg = loss_reg.item()
                    total_loss += loss
                    total_cat_loss += loss_cat
                    total_reg_loss += loss_reg
                else:
                    loss,loss_cat,loss_reg = 0,0,0;
                    total_loss += loss
                    total_cat_loss += loss_cat
                    total_reg_loss += loss_reg

                num_batches += 1
                count += num_examples

                ### classification accuracy
                if model_output_cat.shape[0] == num_examples and model_output_reg.shape[0] == num_examples:
                    _,pred_cat = model_output_cat.max(1)
                    correct = (pred_cat == label).sum().item()
                    total_correct += correct
                    ### regression spread
                    pred_reg = model_output_reg.float()
                    residual_reg = pred_reg - target;
                    sqr_err = residual_reg.square().sum().item()
                    sum_sqr_err += sqr_err
                            
                ### monitor results
                tq.set_postfix({
                    'Loss': '%.5f' % loss,
                    'AvgLoss': '%.5f' % (total_loss / num_batches),
                    'Acc': '%.5f' % (correct / num_examples),
                    'AvgAcc': '%.5f' % (total_correct / count),
                    'MSE': '%.5f' % (sqr_err / num_examples),
                    'AvgMSE': '%.5f' % (sum_sqr_err / count)
                })

                if tb_helper:
                    if tb_helper.custom_fn:
                        with torch.no_grad():
                            tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches,
                                                mode='eval' if for_training else 'test')

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break
                    
    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_counter.items())))
    _logger.info('Eval AvgLoss: %.5f'% (total_loss / num_batches))
    _logger.info('Eval AvgLoss Cat: %.5f'% (total_cat_loss / num_batches))
    _logger.info('Eval AvgLoss Reg: %.5f'% (total_reg_loss / num_batches))
    _logger.info('Eval AvgAccCat: %.5f'%(total_correct / count if count else 0))
    _logger.info('Eval AvgMSE: %.5f'%(sum_sqr_err / count if count else 0))

    if tb_helper:
        tb_mode = 'eval' if for_training else 'test'
        tb_helper.write_scalars([
            ("Loss/%s (epoch)"%(tb_mode), total_loss / num_batches, epoch),
            ("Loss Cat/%s (epoch)"%(tb_mode), total_cat_loss / num_batches, epoch),
            ("Loss Reg/%s (epoch)"%(tb_mode), total_reg_loss / num_batches, epoch),
            ("AccCat/%s (epoch)"%(tb_mode), total_correct / count if count else 0, epoch),
            ("MSE/%s (epoch)"%(tb_mode), sum_sqr_err / count if count else 0, epoch),          
        ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode=tb_mode)
               
    scores_cat = np.concatenate(scores_cat).squeeze()
    scores_reg = np.concatenate(scores_reg).squeeze()

    labels  = {k: _concat(v) for k, v in labels.items()}
    targets = {k: _concat(v) for k, v in targets.items()}
    observers = {k: _concat(v) for k, v in observers.items()}

    _logger.info('Evaluation of metrics\n')
   
    metric_cat_results = evaluate_metrics(labels[data_config.label_names[0]], scores_cat, eval_metrics=eval_cat_metrics)    
    _logger.info('Evaluation Classification metrics: \n%s', '\n'.join(
        ['    - %s: \n%s' % (k, str(v)) for k, v in metric_cat_results.items()]))

    _logger.info('Evaluation of regression metrics\n')
    for idx, (name,element) in enumerate(targets.items()):
        if len(data_config.target_names) == 1:
            metric_reg_results = evaluate_metrics(element, scores_reg, eval_metrics=eval_reg_metrics)
        else:
            metric_reg_results = evaluate_metrics(element, scores_reg[:,idx], eval_metrics=eval_reg_metrics)

        _logger.info('Evaluation Regression metrics for '+name+' target: \n%s', '\n'.join(
            ['    - %s: \n%s' % (k, str(v)) for k, v in metric_reg_results.items()]))        

    if for_training:
        return total_loss / num_batches;
    else:
        if scores_reg.ndim and scores_cat.ndim: 
            scores_reg = scores_reg.reshape(len(scores_reg),len(data_config.target_names))
            scores = np.concatenate((scores_cat,scores_reg),axis=1)
            return total_loss / num_batches, scores, labels, targets, observers
        else:
            return total_loss / num_batches, scores_reg, labels, targets, observers;

class TensorboardHelper(object):

    def __init__(self, tb_comment, tb_custom_fn):
        self.tb_comment = tb_comment
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(comment=self.tb_comment)
        _logger.info('Create Tensorboard summary writer with comment %s' % self.tb_comment)

        # initiate the batch state
        self.batch_train_count = 0

        # load custom function
        self.custom_fn = tb_custom_fn
        if self.custom_fn is not None:
            from weaver.utils.import_tools import import_module
            from functools import partial
            self.custom_fn = import_module(self.custom_fn, '_custom_fn')
            self.custom_fn = partial(self.custom_fn.get_tensorboard_custom_fn, tb_writer=self.writer)

    def __del__(self):
        self.writer.close()

    def write_scalars(self, write_info):
        for tag, scalar_value, global_step in write_info:
            self.writer.add_scalar(tag, scalar_value, global_step)
