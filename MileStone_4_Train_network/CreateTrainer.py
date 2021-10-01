from ignite.engine import Engine, Events
from pathlib import Path
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import ModelCheckpoint, Checkpoint, DiskSaver
from ignite.metrics import Accuracy, Loss, RunningAverage
import torch


def create_trainer(model, optimizer, criterion, lr_scheduler, config):
    device = config['device']
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

    @trainer.on(Events.ITERATION_COMPLETED(every=10))
    def save_checkpoint():
        fp = Path(config.get("output_path", "./output")) / "checkpoint.pt"
        torch.save(model.state_dict(), fp)

    # @trainer.on(Events.EPOCH_COMPLETED)
    # def log_training_results(engine):
    #     train_evaluator.run(train_loader)
    #     metrics = train_evaluator.state.metrics
    #     avg_accuracy = metrics['accuracy']
    #     avg_nll_loos = metrics['NLLLoss']
    #     pbar.log_message(
    #         "Training Results - Epoch: {} Avg accuracy: {:.2f} avg loss {:.2f}".format(engine.state.epoch, avg_accuracy,
    #                                                                                    avg_nll_loos)
    #     )

    # @trainer.on(Events.EPOCH_COMPLETED)
    # def log_validation_results(engine):
    #     validation_evaluator.run(valid_loader)
    #     metrics = validation_evaluator.state.metrics
    #     avg_accuracy = metrics['accuracy']
    #     avg_nll_loos = metrics['NLLLoss']
    #     pbar.log_message
    #     (
    #         "Training Results - Epoch: {} Avg accuracy: {:.2f} avg loss {:.2f}".format(
    #             engine.state.epoch, avg_accuracy, avg_nll_loos)
    #     )
    #     pbar.n = pbar.last_print_n = 0

    # Add progress bar showing batch loss value
    Accuracy(output_transform=lambda x: x).attach(train_evaluator, 'train_acc')
    Loss(criterion).attach(train_evaluator, 'train_NLLLoss')

    Accuracy(output_transform=lambda x: x).attach(
        validation_evaluator, 'val_acc')
    Loss(criterion).attach(validation_evaluator, 'val_NLLLoss')

    RunningAverage(output_transform=lambda x: x).attach(
        trainer, 'loss')

    checkpointer = ModelCheckpoint(
        config.get('checkpointPath', './tmp/models'), config.get('model',
                                                                 'Unknown_you_lazy_bastart'),
        n_saved=config.get('max_saved_checkout', 2), create_dir=True,
        save_as_state_dict=True, require_empty=config.get('require_empty', True)
    )

    pbar = ProgressBar(persist=True, bar_format="")
    pbar.attach(trainer, ['loss'])
    pbar.attach(trainer, output_transform=lambda x: {"batch loss": x})

    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              checkpointer, {config.get('model'): model}
                              )
    to_save = {'trainer': trainer,
               'model': model,
               'optimizer': optimizer,
               'lr_scheduler': lr_scheduler,
               }
    
    handler = Checkpoint(to_save, DiskSaver(
        config['trainer_save_path'], create_dir=True))
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)

    return trainer, train_evaluator, validation_evaluator, test_evaluator, pbar
