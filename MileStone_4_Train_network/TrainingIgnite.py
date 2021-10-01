from InitializeModel import initializeModel
from CreateTrainer import create_trainer
from getData import getData
#from ignite.engine import create_supervised_evaluator, Events
# from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.contrib.engines import common
# from ignite.handlers import EarlyStopping
# from ignite.metrics import ClassificationReport
import ignite.distributed as idist
# import torch

# source https://colab.research.google.com/github/pytorch/ignite/blob/master/assets/tldr/teaser.ipynb#scrollTo=dFglXKeKOgdW
# dependenciues

# ignite

# PyTorch


def training(config):
    # Setup dataflow and
    data = getData(config)
    model, optimizer, criterion, lr_scheduler = initializeModel(config)
    model.to(config.get('device', 'cpu'))
    # Setup model trainer and evaluator
    trainer, train_evaluator, validation_evaluator, test_evaluator, evaluator, pbar = create_trainer(
        model, optimizer, criterion, lr_scheduler, config, data
    )
    data_loader, _, _ = data
    train_loader, train_sampler = data_loader

    if config['train_now'] is not True:
        return trainer, model, optimizer, criterion, lr_scheduler, train_loader

    if idist.get_rank() == 0:
        # print('Start logging', config['tensorboard_logger_path'])
        tb_logger = common.setup_tb_logging(
            output_path=config.get("tensorboard_logger_path", "output"),
            trainer=trainer, optimizers=optimizer,
            evaluators={"validation": evaluator,
                        # 'cr': evaluator,
                        # 'loss': trainer,
                        'val_NLLLoss': validation_evaluator,
                        # 'val_acc': validation_evaluator,
                        'train_NLLLoss': train_evaluator,
                        # 'train_acc': train_evaluator,
                        },
            log_every_iters=1,
        )

    trainer.run(train_loader, max_epochs=config.get("max_epochs", 3))

    if idist.get_rank() == 0:
        # print("cloosing tb_logger...")
        tb_logger.close()
    # tb_logger.close()
    # return trainer, (train_loader, train_sampler), (valid_loader,
    #                                                 valid_sampler), (test_loader, test_sampler)
