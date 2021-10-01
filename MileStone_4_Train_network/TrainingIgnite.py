from InitializeModel import initializeModel
from CreateTrainer import create_trainer
from getData import getData
from ignite.engine import create_supervised_evaluator, Events
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.contrib.engines import common
from ignite.metrics.precision import Precision
from ignite.metrics.recall import Recall
from ignite.handlers import EarlyStopping
from ignite.metrics import ClassificationReport
from ignite.contrib.engines.common import save_best_model_by_val_score
import ignite.distributed as idist
import torch
# source https://colab.research.google.com/github/pytorch/ignite/blob/master/assets/tldr/teaser.ipynb#scrollTo=dFglXKeKOgdW
# dependenciues

# ignite

# PyTorch


def training(config):
    # Setup dataflow and
    (train_loader, train_sampler), (valid_loader,
                                    valid_sampler), (test_loader, test_sampler) = getData(config)

    model, optimizer, criterion, lr_scheduler = initializeModel(config)

    model.to(config.get('device', 'cpu'))

    # Setup model trainer and evaluator
    trainer, train_evaluator, validation_evaluator, test_evaluator, pbar = create_trainer(
        model, optimizer, criterion, lr_scheduler, config)
    if config['train_now'] is not True:
        return trainer, model, optimizer,criterion ,lr_scheduler, train_loader

    def thresholded_output_transform(output):
        y_pred, y = output
        y_pred = torch.round(y_pred)
        return y_pred, y

    # alpha = 0.98
    # acc_metric = RunningAverage(Accuracy())
    # # acc_metric.attach(trainer, 'running_avg_accuracy')

    evaluator = create_supervised_evaluator(
        model,
        metrics={"accuracy": Accuracy(),
                 "loss": RunningAverage(Accuracy()),
                 "precision": Precision(average=False),
                 'recall': Recall(average=False),
                 'cr': ClassificationReport(output_dict=True, is_multilabel=False)
                 },
        device=config.get('device', 'cpu')
    )

    # Run model evaluation every 3 epochs and show results
    @trainer.on(Events.EPOCH_COMPLETED(every=1))
    def evaluate_model():
        state = evaluator.run(valid_loader)
        metrics = state.metrics
        acc = metrics['accuracy']
        loss = metrics['loss']
        #recAll = metrics['recall']
        #preciss = metrics['precision']
        #F1 = preciss * recAll * 2 / (preciss + recAll + 1e-20)
        #F1 = MetricsLambda(lambda t: torch.mean(t).item(), F1)
        res_dic = metrics['cr']
        #res_dic = ast.literal_eval(res)
        pbar.log_message(
            "Validation set: Average loss: {:.4f}, Accuracy: {:.4f} F1 for class {} is {:.4f}".format(
                loss, acc, 14, res_dic['14']['f1-score'])
        )
        train_evaluator.run(train_loader)
        validation_evaluator.run(valid_loader)

        # if idist.get_rank() == 0:
        #     print(state.metrics)

    # @trainer.on(Events.EPOCH_COMPLETED(every=1))
    # def log_training_results(engine):
    #     train_evaluator.run(train_loader)
        # metrics = train_evaluator.state.metrics
        # avg_accuracy = metrics['train_acc']
        # avg_nll_loos = metrics['train_NLLLoss']
        # pbar.log_message(
        #     "Training Results - Epoch: {} Avg accuracy: {:.2f} avg loss {:.2f}".format(engine.state.epoch, avg_accuracy,
        #                                                                                avg_nll_loos)
        # )

    # @trainer.on(Events.EPOCH_COMPLETED(every=1))
    # def log_validation_results(engine):
        # validation_evaluator.run(valid_loader)
    #     metrics = validation_evaluator.state.metrics
    #     avg_accuracy = metrics['val_acc']
    #     avg_nll_loos = metrics['val_NLLLoss']
    #     pbar.log_message
    #     (
    #         "Validation Results - Epoch: {} Avg accuracy: {:.2f} avg loss {:.2f}".format(
    #             engine.state.epoch, avg_accuracy, avg_nll_loos)
    #     )
    #     pbar.n = pbar.last_print_n = 0

    # @ trainer.on(Events.COMPLETED)
    # def score_function(engine):
    #     validation_evaluator.run(valid_loader)
    #     metrics = validation_evaluator.state.metrics
    #     val_loss = metrics['val_NLLLoss']
    #     return -val_loss

    # handler_earlyStopping = EarlyStopping(
        # patience=5, score_function=score_function, trainer=trainer)
    # validation_evaluator.add_event_handler(
        # Events.COMPLETED, handler_earlyStopping)

    # tb_logger = common.setup_tb_logging(
    #     config.get("output_path", "output"), trainer, optimizer, evaluators={"validation": evaluator},
    # )

    # Setup tensorboard experiment tracking
    if idist.get_rank() == 0:
        # print('Start logging', config['tensorboard_logger_path'])
        tb_logger = common.setup_tb_logging(
            output_path=config.get("tensorboard_logger_path", "output"),
            trainer=trainer, optimizers=optimizer,
            evaluators={"validation": evaluator,
                        # 'cr': evaluator,
                        'val_NLLLoss': validation_evaluator,
                        # 'val_acc': validation_evaluator,
                        'train_NLLLoss': train_evaluator,
                        # 'train_acc': train_evaluator,
                        },
            log_every_iters=1,
        )
    bestModelSaver = save_best_model_by_val_score(output_path=config['best_model_path'],
                                                  evaluator=evaluator, model=model,
                                                  metric_name='accuracy',
                                                  n_saved=config.get(
                                                      "best_n_saved", 2)
                                                  )
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, bestModelSaver)

    trainer.run(train_loader, max_epochs=config.get("max_epochs", 3))

    if idist.get_rank() == 0:
        # print("cloosing tb_logger...")
        tb_logger.close()
    # tb_logger.close()
    # return trainer, (train_loader, train_sampler), (valid_loader,
    #                                                 valid_sampler), (test_loader, test_sampler)
