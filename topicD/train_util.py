# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import os
import logging
import datetime
import yaml
import torch
import numpy as np
import tableprint as tp
from pprint import pformat


def genlogger(outdir, fname, level):
    formatter = logging.Formatter(
        "[ %(levelname)s : %(asctime)s ] - %(message)s")
    logging.basicConfig(
        filemode="w",
        level=getattr(logging, level),
        format="[ %(levelname)s : %(asctime)s ] - %(message)s")
    logger = logging.getLogger(__name__ + "." + fname)
    # Dump log to file
    fh = logging.FileHandler(os.path.join(outdir, fname))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def parse_config_or_kwargs(config_file, **kwargs):
    with open(config_file) as con_read:
        yaml_config = yaml.load(con_read, Loader=yaml.FullLoader)
    # passed kwargs will override yaml config
    return dict(yaml_config, **kwargs)


def criterion_improver(mode):
    assert mode in ("loss", "acc")
    best_value = np.inf if mode == "loss" else 0

    def comparator(x, best_x):
        return x < best_x if mode == "loss" else x > best_x

    def inner(x):
        nonlocal best_value

        if comparator(x, best_value):
            best_value = x
            return True
        return False
    return inner


def on_training_started(engine, logger, header):
    logger.info("<== Training Started ==>")
    for line in tp.header(header, style="grid").split("\n"):
        logger.info(line)


def log_results(engine,
                cv_evaluator, 
                cv_dataloader, 
                logger,
                train_metric_keys=["loss", "accuracy"], 
                cv_metric_keys=["loss", "accuracy"]):
    train_metrics = engine.state.metrics
    cv_evaluator.run(cv_dataloader)
    cv_metrics = cv_evaluator.state.metrics
    line = (engine.state.epoch, )
    for key in train_metric_keys:
        line += (train_metrics[key], )
    for key in cv_metric_keys:
        line += (cv_metrics[key], )
    logger.info(tp.row(line, style="grid"))


def save_model_on_improved(engine,
                           criterion_improved, 
                           model, 
                           config_params,
                           save_path):
    if criterion_improved(engine.state.metrics["loss"]):
        torch.save({
            "model": model,
            "config": config_params
        }, save_path)


def on_training_ended(engine, logger, n):
    logger.info(tp.bottom(n, style="grid"))


def update_reduce_on_plateau(engine, scheduler):
    cv_loss = engine.state.metrics["loss"]
    scheduler.step(cv_loss)
