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
from ignite.metrics import Loss, RunningAverage, Accuracy
from ignite.handlers import ModelCheckpoint, EarlyStopping

sys.path.append("/mnt/lustre/sjtu/home/zkz01/leinao")
from models import *
from datasets.dataset import create_dataloader

sys.path.append("/mnt/lustre/sjtu/home/xnx98/utils/")
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
        shuffle=False,
        transform=scaler.transform,
        **config_parameters["dataloader_args"]
    )
    return train_dataloader, dev_dataloader, test_dataloader


def getmodel(config_parameters):
    encoder = eval(config_parameters["encoder"])(
        input_dim=config_parameters["num_freq"]*(2*config_parameters["context"]+1),
        **config_parameters["encoder_args"])
    encoder = model_init(encoder,config_parameters)
    classifier = eval(config_parameters["classifier"])(
        input_dim=config_parameters["encoder_args"]["latent_dim"],
        **config_parameters["classifier_args"])
    return encoder, classifier

def model_init(encoder,config_parameters):
    path = config_parameters['encoder_path']
    if path is None:
        return encoder
    else:
        auto_encoder = torch.load(path)
        encoder_dict = encoder.state_dict()
        pretrained_dict = auto_encoder['model'].state_dict().items()
        pretrained_dict = {k[8:]:v for k,v in pretrained_dict if k.startswith("encoder.")}
        encoder_dict.update(pretrained_dict)
        encoder.load_state_dict(encoder_dict)
        return encoder

def prepare_batch(batch, device=None):
    feats, labels = batch
    feats = feats.float()
    labels = labels.long()
    if device:
        feats = feats.to(device)
        labels = labels.to(device)
    return feats, labels


def create_trainer(encoder, classifier, optimizer, loss_fn,fix_encoder, device=None, metrics={}):
    if device:
        encoder.to(device)
        classifier.to(device)
    if fix_encoder == True or fix_encoder == 'true':
        for par in encoder.parameters():
            par.requires_grad = False
    def _update(engine, batch):
        if fix_encoder == True or fix_encoder == 'true':
            encoder.eval()
        else:
            encoder.train()
        # encoder.train()
        classifier.train()
        optimizer.zero_grad()
        feats, labels = prepare_batch(batch, device)
        embeddings = encoder(feats)
        probs = classifier(embeddings)
        loss = loss_fn(probs, labels)
        loss.backward()
        optimizer.step()
        return loss.item()

    engine = Engine(_update)
    
    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def create_evaluator(encoder, classifier, device=None, metrics={}):
    if device:
        encoder.to(device)
        classifier.to(device)

    def _inference(engine, batch):
        encoder.eval()
        classifier.eval()

        with torch.no_grad():
            feats, labels = prepare_batch(batch, device)
            embeddings = encoder(feats)
            probs = classifier(embeddings)
            return probs, labels
    
    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(config, outputdir=None, **kwargs):
    np.random.seed(777)
    torch.manual_seed(777)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(777) # fix random seed
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
    encoder, classifier = getmodel(config_parameters)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    logger.info("<== Encoder ==>")
    for line in pformat(encoder).split("\n"):
        logger.info(line)
    logger.info("<== Classifier ==>")
    for line in pformat(classifier).split("\n"):
        logger.info(line)

    optimizer = getattr(torch.optim, config_parameters["optimizer"])(
        list(encoder.parameters()) + list(classifier.parameters()),
        **config_parameters["optimizer_args"])
    scheduler = getattr(torch.optim.lr_scheduler, config_parameters["scheduler"])(
        optimizer, **config_parameters["scheduler_args"])

    criterion_improved = train_util.criterion_improver(config_parameters["improvecriterion"])

    trainer = create_trainer(encoder,
                             classifier,
                             optimizer,
                             loss_fn,
                             config_parameters["fix_encoder"],
                             device)
    evaluator = create_evaluator(encoder,
                                 classifier,
                                 device,
                                 metrics={"loss": Loss(loss_fn),
                                          "accuracy": Accuracy()})

    RunningAverage(output_transform=lambda x: x).attach(trainer, "run_loss")
    pbar = ProgressBar(persist=False)
    pbar.attach(trainer)

    trainer.add_event_handler(Events.STARTED,
                              train_util.on_training_started,
                              logger,
                              ["Epoch", "Loss(CV)", "Acc(CV)"])
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              train_util.log_results,
                              evaluator,
                              cv_dataloader,
                              logger,
                              train_metric_keys=[])
    trainer.add_event_handler(Events.COMPLETED,
                              train_util.on_training_ended,
                              logger, 3)
    evaluator.add_event_handler(Events.EPOCH_COMPLETED,
                                train_util.save_model_on_improved,
                                criterion_improved,
                                {"encoder": encoder, "classifier": classifier},
                                config_parameters,
                                save_path)

    ckpt_handler = ModelCheckpoint(outputdir,
                                   "exp",
                                   save_interval=2,
                                   n_saved=2,
                                   require_empty=False,
                                   save_as_state_dict=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, ckpt_handler, {"encoder": encoder, "classifier": classifier})
    
    early_stop_handler = EarlyStopping(patience=config_parameters["early_stop"],
                                       score_function=lambda engine: -engine.state.metrics["loss"],
                                       trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, early_stop_handler)

    @trainer.on(Events.COMPLETED, eval_dataloader, save_path, loss_fn, logger)
    def eval_on_test(engine, dataloader, save_path, loss_fn, logger):
        saved_model = torch.load(save_path)["model"]
        evaluator = create_evaluator(saved_model["encoder"],
                                     saved_model["classifier"],
                                     metrics={"loss": Loss(loss_fn),
                                              "accuracy": Accuracy()},
                                     device=device)
        evaluator.run(dataloader)
        logger.info("Acc on Eval: {:.2%}".format(evaluator.state.metrics["accuracy"]))

    trainer.run(train_dataloader, max_epochs=config_parameters["epochs"])


if __name__ == "__main__":
    fire.Fire(main)

