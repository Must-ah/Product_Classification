import torch
from ignite.contrib.engines.common import save_best_model_by_val_score
from ignite.metrics import Accuracy, Loss, RunningAverage, ClassificationReport
from ignite.handlers import ModelCheckpoint, Checkpoint, DiskSaver
from ignite.contrib.handlers import ProgressBar
# from ignite.metrics.precision import Precision
# from ignite.metrics.recall import Recall

from pathlib import Path
from ignite.engine import Engine, Events, create_supervised_evaluator
# source https://colab.research.google.com/github/pytorch/ignite/blob/master/assets/tldr/teaser.ipynb#scrollTo=dFglXKeKOgdW


def create_trainer(model, optimizer, criterion, lr_scheduler, config, data):
    device = config['device']
    (train_loader, train_sampler), (valid_loader,
                                    valid_sampler), (test_loader, test_sampler) = data
    # Define any training logic for iteration update

    def train_step(engine, batch):
        x, y = batch[0].to(device), batch[1].to(device)
        model.train()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        return loss.item()

    def eval_model_on_batch(engine, batch):
        """
        Evaluation of the model on a single batch

        Args:
            engine: ignite.engine.Engine
            batch: tuple contains the training sample and their labels

        """
        model.eval()
        with torch.no_grad():
            data, target = batch
            y_pred = model(data.to(device))
            return y_pred, target.to(device)

    trainer = Engine(train_step)
    train_evaluator = Engine(eval_model_on_batch)
    validation_evaluator = Engine(eval_model_on_batch)
    test_evaluator = Engine(eval_model_on_batch)

    evaluator = create_supervised_evaluator(
        model,
        metrics={"accuracy": Accuracy(),
                 "loss": RunningAverage(Accuracy()),
                 #  "precision": Precision(average=False),
                 #  'recall': Recall(average=False),
                 'cr': ClassificationReport(output_dict=True, is_multilabel=False)
                 },
        device=config.get('device', 'cpu')
    )

    @trainer.on(Events.ITERATION_COMPLETED(every=10))
    def save_checkpoint():
        fp = Path(config.get("output_path", "./output")) / "checkpoint.pt"
        torch.save(model.state_dict(), fp)

    pbar = ProgressBar(persist=True, bar_format="")

    # Run model evaluation every 3 epochs and show results
    @trainer.on(Events.EPOCH_COMPLETED(every=1))
    def evaluate_model():
        state = evaluator.run(valid_loader)
        metrics = state.metrics
        acc = metrics['accuracy']
        loss = metrics['loss']
        # recAll = metrics['recall']
        # preciss = metrics['precision']
        # F1 = preciss * recAll * 2 / (preciss + recAll + 1e-20)
        # F1 = MetricsLambda(lambda t: torch.mean(t).item(), F1)
        res_dic = metrics['cr']
        # res_dic = ast.literal_eval(res)
        train_evaluator.run(train_loader)
        validation_evaluator.run(valid_loader)

        pbar.log_message(
            "Validation set: Average loss: {:.4f}, Accuracy: {:.4f} F1 for class {} is {:.4f}".format(
                loss, acc, 14, res_dic['14']['f1-score'])
        )

    RunningAverage(output_transform=lambda x: x).attach(
        trainer, 'loss'
    )
    pbar.attach(trainer, ['loss'])
    pbar.attach(trainer, output_transform=lambda x: {"batch loss": x})

    Accuracy(output_transform=lambda x: x).attach(train_evaluator, 'train_acc')
    Loss(criterion).attach(train_evaluator, 'train_NLLLoss')

    Accuracy(output_transform=lambda x: x).attach(
        validation_evaluator, 'val_acc')
    Loss(criterion).attach(validation_evaluator, 'val_NLLLoss')

    checkpointer = ModelCheckpoint(
        config.get('checkpointPath', './tmp/models'), config.get('model',
                                                                 'Unknown_you_lazy_bastart'),
        n_saved=config.get('max_saved_checkout', 2), create_dir=True,
        save_as_state_dict=True, require_empty=config.get('require_empty', True)
    )

    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=3),
                              checkpointer, {config.get('model'): model}
                              )
    to_save = {'trainer': trainer,
               'model': model,
               'optimizer': optimizer,
               'lr_scheduler': lr_scheduler,
               }

    handler = Checkpoint(to_save,
                         DiskSaver(config['trainer_save_path'],
                                   create_dir=True, require_empty=config['require_empty']
                                   )
                         )
    bestModelSaver = save_best_model_by_val_score(output_path=config['best_model_path'],
                                                  evaluator=evaluator, model=model,
                                                  metric_name='accuracy',
                                                  n_saved=config.get(
                                                      "best_n_saved", 2)
                                                  )
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, bestModelSaver)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)

    return trainer, train_evaluator, validation_evaluator, test_evaluator, evaluator, pbar
