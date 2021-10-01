import sys
sys.path.insert(0, '../')
from torch.nn import NLLLoss
from torch import optim
from MileStone_3_Create_Neural_Network.WillysShuffleNet_v2_x1_0 import myShuffleNet_v2_x1_0
from MileStone_3_Create_Neural_Network.WillyMnasNet0_5 import MyMnasNet0_5
from MileStone_3_Create_Neural_Network.WillyNet import WillyNet
from MileStone_3_Create_Neural_Network.WillyAlexNet import MyAlexNet

# Neural Networks


def initializeModel(config):
    # ToDo return model from my private zoo
    modelName = config["model"]
    if modelName == "MyAlexNet":
        model = MyAlexNet()
    elif modelName == "WillyNet":
        model = WillyNet()
    elif modelName == "WillyMnasNet0_5":
        model = MyMnasNet0_5()
    elif modelName == "WillysShuffleNet_v2_x1_0":
        model = myShuffleNet_v2_x1_0()
    else:
        raise ValueError(f"Model {modelName} not supported or whatever something else")

    # model = WillyNet()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.get("lr", 1e-2),
        momentum=config.get("momentum", 0.5),
        weight_decay=config.get('weight_decay', 1e-6),
        nesterov=True,
    )
    criterion = NLLLoss()
    le = config['num_iters_per_epoch']
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                   step_size=le,
                                                   gamma=config.get(
                                                       'gamma', 0.9)
                                                   )
    return model, optimizer, criterion, lr_scheduler
