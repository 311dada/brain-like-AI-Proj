import os
import sys
import datetime
from pprint import pformat

import numpy as np
import torch
import yaml
import fire
from sklearn import preprocessing
from ignite.contrib.handlers import ProgressBar
from ignite.engine.engine import Engine, Events
from ignite.metrics import Loss, RunningAverage
from ignite.handlers import ModelCheckpoint, EarlyStopping

from models import *
from datasets.dataset import create_dataloader

import train_util


def getdataloaders(config_parameters):
    context = config_parameters["context"]
    
    if os.path.exists(config_parameters["scaler_path"]):
        scaler = torch.load(config_parameters["scaler_path"])
    else:
        scaler = preprocessing.StandardScaler()
        for feat, _ in create_dataloader(config_parameters["train_feat_stream"].format(context, context),
                                         config_parameters["train_label_stream"]):
            scaler.partial_fit(feat)
        torch.save(scaler, config_parameters["scaler_path"])
    train_dataloader = create_dataloader(
        config_parameters["train_feat_stream"].format(context, context),
        config_parameters["train_label_stream"],
        shuffle=True,
        transform=scaler.transform,
        **config_parameters["dataloader_args"]
    )
    dev_dataloader = create_dataloader(
        config_parameters["dev_feat_stream"].format(context, context),
        config_parameters["dev_label_stream"],
        shuffle=False,
        transform=scaler.transform,
        **config_parameters["dataloader_args"]
    )
    test_dataloader = create_dataloader(
        config_parameters["test_feat_stream"].format(context, context),
        config_parameters["test_label_stream"],
        shuffle=True,
        transform=scaler.transform,
        **config_parameters["dataloader_args"]
    )
    return train_dataloader, dev_dataloader, test_dataloader


def getmodel(config_parameters):
    encoder = eval(config_parameters["encoder"])(
        input_dim=config_parameters["num_freq"] * (2 * config_parameters["context"] + 1),
        **config_parameters["encoder_args"])
    decoder = eval(config_parameters["decoder"])(
        out_dim=config_parameters["num_freq"] * (2 * config_parameters["context"] + 1),
        **config_parameters["decoder_args"])
    return eval(config_parameters["model"])(encoder, decoder)


def prepare_batch(batch, device=None):
    feats, labels = batch
    feats = feats.float()
    labels = labels.long()
    if device:
        feats = feats.to(device)
    return feats, labels


def create_trainer(model, optimizer, loss_fn, device=None, metrics={}):
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        feats, labels = prepare_batch(batch, device)
        feats_hat = model(feats)
        loss = loss_fn(feats, feats_hat)
        loss.backward()
        optimizer.step()
        return feats, feats_hat

    engine = Engine(_update)
    
    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def create_evaluator(model, device=None, metrics={}):
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            feats, labels = prepare_batch(batch, device)
            feats_hat = model(feats)
            return feats, feats_hat
    
    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(config, outputdir=None, **kwargs):
    config_parameters = train_util.parse_config_or_kwargs(config, **kwargs)
    
    if outputdir is None:
        outputdir = os.path.join("exp", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))

    try:
        os.makedirs(outputdir)
    except IOError:
        pass
    save_path = os.path.join(outputdir, "model.pth")

    logger = train_util.genlogger(outputdir, "train.log", "INFO")
    logger.info("Storing data at: {}".format(outputdir))
    logger.info("<== Passed Arguments ==>")
    for line in pformat(config_parameters).split("\n"):
        logger.info(line)

    train_dataloader, cv_dataloader, eval_dataloader = getdataloaders(config_parameters)
    model = getmodel(config_parameters)
    
    loss_fn = torch.nn.MSELoss()
    
    logger.info("<== Model ==>")
    for line in pformat(model).split("\n"):
        logger.info(line)

    optimizer = getattr(torch.optim, config_parameters["optimizer"])(
        model.parameters(), **config_parameters["optimizer_args"])
    scheduler = getattr(torch.optim.lr_scheduler, config_parameters["scheduler"])(
        optimizer, **config_parameters["scheduler_args"])

    criterion_improved = train_util.criterion_improver(config_parameters["improvecriterion"])

    trainer = create_trainer(model, optimizer, loss_fn, device, metrics={"loss": Loss(loss_fn)})
    evaluator = create_evaluator(model, device, metrics={"loss": Loss(loss_fn)})

    pbar = ProgressBar(persist=False)
    pbar.attach(trainer)

    trainer.add_event_handler(Events.STARTED,
                              train_util.on_training_started,
                              logger,
                              ["Epoch", "Loss(T)", "Loss(CV)"])
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              train_util.log_results,
                              evaluator,
                              cv_dataloader,
                              logger,
                              train_metric_keys=["loss"],
                              cv_metric_keys=["loss"])
    evaluator.add_event_handler(Events.EPOCH_COMPLETED,
                              train_util.save_model_on_improved,
                              criterion_improved,
                              model,
                              config_parameters,
                              save_path)
    trainer.add_event_handler(Events.COMPLETED,
                              train_util.on_training_ended,
                              logger, 3)

    ckpt_handler = ModelCheckpoint(outputdir,
                                   "exp",
                                   save_interval=2,
                                   n_saved=2,
                                   require_empty=False,
                                   save_as_state_dict=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, ckpt_handler, {"model": model})
    
    early_stop_handler = EarlyStopping(patience=config_parameters["early_stop"],
                                       score_function=lambda engine: -engine.state.metrics["loss"],
                                       trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, early_stop_handler)

    @trainer.on(Events.COMPLETED, eval_dataloader, save_path, loss_fn, logger)
    def eval_on_test(engine, dataloader, save_path, loss_fn, logger):
        model = torch.load(save_path)["model"]
        evaluator = create_evaluator(model,
                                     metrics={"loss": Loss(loss_fn)},
                                     device=device)
        evaluator.run(dataloader)
        logger.info("Loss on Eval: {:.2f}".format(evaluator.state.metrics["loss"]))

    trainer.run(train_dataloader, max_epochs=config_parameters["epochs"])


if __name__ == "__main__":
    fire.Fire(main)

