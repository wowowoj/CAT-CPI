import os
import torch
import argparse
from datetime import datetime
from utils import data_loader, get_pic_path
from CAT_model import CAT, Train_model, Tester
from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve, auc, average_precision_score

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def model_run(args):
    # time
    ISOTIMEFORMAT = '%Y_%m%d_%H%M'
    run_time = datetime.now().strftime(ISOTIMEFORMAT)

    print("dataset_name: ", args.dataset_name, run_time)
    resul_name = "result/" + args.dataset_name
    if not os.path.exists(resul_name):
        os.makedirs(resul_name)

    file_AUCs_test = resul_name + "/" + \
                     run_time + " " + \
                     args.dataset_name + " " + \
                     str(args.batch_size) + " " + \
                     str(args.lr) + " " + \
                     str(args.lr_decay) + " " + \
                     str(args.weight_decay) + " " + \
                     str(args.mlp_flag) + " .txt"

    # ********************************* Train_dataset *********************************
    train_pic_path = "data/" + args.dataset_name + "/train/" + "Pic_" + str(args.pic_size) + "_" + str(
        args.pic_size) + "/pic_inf_data"
    train_pic = get_pic_path(train_pic_path)
    train_protein_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_train_proteins.npy"
    train_itr_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_train_interactions.npy"
    train_dataset, train_loader = data_loader(args.batch_size, train_pic, train_protein_name, train_itr_name)

    print(train_dataset)

    # ********************************* Test_dataset *********************************
    test_pic_path = "data/" + args.dataset_name + "/test/" + "Pic_" + str(args.pic_size) + "_" + str(
        args.pic_size) + "/pic_inf_data"
    test_pic = get_pic_path(test_pic_path)
    test_protein_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_test_proteins.npy"
    test_itr_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_test_interactions.npy"
    test_dataset, test_loader = data_loader(args.batch_size, test_pic, test_protein_name, test_itr_name)

    # ********************************* Val_dataset *********************************
    val_pic_path = "data/" + args.dataset_name + "/val/" + "Pic_" + str(args.pic_size) + "_" + str(
        args.pic_size) + "/pic_inf_data"
    val_pic = get_pic_path(val_pic_path)
    val_protein_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_val_proteins.npy"
    val_itr_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_val_interactions.npy"
    val_dataset, val_loader = data_loader(args.batch_size, val_pic, val_protein_name, val_itr_name)

    torch.manual_seed(2)
    model = CAT(embed_dim=args.embed_dim,
                depth=args.depth,
                drop_ratio=args.drop,
                usemlp=args.mlp_flag
                ).to(device)

    decay_interval = args.decay_interval
    lr, lr_decay, weight_decay = map(float, [args.lr, args.lr_decay, args.weight_decay])
    trainer = Train_model(model, lr, weight_decay)

    for epoch in range(1, args.epochs + 1):
        print("training  Epoch: " + str(epoch))
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay
        total_loss = []
        for i, data_train in enumerate(train_loader):
            if data_train[0].shape[0] <= 1:
                break
            loss_train = trainer.train(data_train)  #
            total_loss.append(loss_train)
            if (i + 1) % 50 == 0:
                print(
                    "Training [Epoch %d/%d] [Batch %d/%d] [batch_size %d] [loss_train : %f]"
                    % (epoch, args.epochs, i, len(train_loader), data_train[0].shape[0], loss_train)
                )
        model_path = "data/" + args.dataset_name + "/output/model/"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        file_model = model_path + run_time + "_" + str(epoch) + ".model"
        trainer.save_model(model, file_model)
        print("avg loss:", sum(total_loss) / len(train_loader))
        with torch.no_grad():
            val(args, file_model, val_loader)
            test(args, file_model, test_dataset, test_loader, file_AUCs_test)
    return model


def val(args, file_model, val_loader):
    torch.manual_seed(2)
    model = torch.load(file_model)
    valer = Tester(model)
    Loss, y_label, y_pred, y_score = [], [], [], []
    for i, data_list in enumerate(val_loader):
        loss, correct_labels, predicted_labels, predicted_scores = valer.test(data_list)
        Loss.append(loss)
        for c_l in correct_labels:
            y_label.append(c_l)
        for p_l in predicted_labels:
            y_pred.append(p_l)
        for p_s in predicted_scores:
            y_score.append(p_s)

    loss_val = sum(Loss) / len(val_loader)
    AUC_val = roc_auc_score(y_label, y_score)
    AUPRC = average_precision_score(y_label, y_score)

    precision_val = precision_score(y_label, y_pred)
    recall_val = recall_score(y_label, y_pred)
    f1_score = (2 * precision_val * recall_val) / (recall_val + precision_val + 0.0001)
    print(
        "Valing  batch_size %d  [loss : %.3f] [AUC : %.3f] [AUPRC : %.3f] [precision : %.3f] [recall : %.3f] [F1 : %.3f] "
        % (args.batch_size, loss_val, AUC_val, AUPRC, precision_val, recall_val, f1_score)
    )


def test(args, file_model, test_dataset, test_loader, file_AUCs_test):
    torch.manual_seed(2)
    model = torch.load(file_model)
    tester = Tester(model)
    Loss, y_label, y_pred, y_score = [], [], [], []
    for i, data_list in enumerate(test_loader):
        loss, correct_labels, predicted_labels, predicted_scores = tester.test(data_list)
        Loss.append(loss)
        for c_l in correct_labels:
            y_label.append(c_l)
        for p_l in predicted_labels:
            y_pred.append(p_l)
        for p_s in predicted_scores:
            y_score.append(p_s)

    loss_test = sum(Loss) / len(test_loader)
    AUC_test = roc_auc_score(y_label, y_score)
    AUPRC = average_precision_score(y_label, y_score)

    precision_test = precision_score(y_label, y_pred)
    recall_test = recall_score(y_label, y_pred)
    f1_score = (2 * precision_test * recall_test) / (recall_test + precision_test + 0.0001)
    print(
        "Testing  batch_size %d  [loss : %.3f] [AUC : %.3f] [AUPRC : %.3f] [precision : %.3f] [recall : %.3f] [F1 : %.3f] "
        % (args.batch_size, loss_test, AUC_test, AUPRC, precision_test, recall_test, f1_score)
    )
    print()
    AUCs = [len(test_dataset),
            len(test_loader),
            format(loss_test, '.3f'),
            format(AUC_test, '.3f'),
            format(precision_test, '.3f'),
            format(recall_test, '.3f'),
            format(f1_score, ".3f"),
            format(AUPRC, '.3f')]
    tester.save_AUCs(AUCs, file_AUCs_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="Daivs")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--pic_size', type=int, default=128)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--drop', type=float, default=0.)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument("--decay_interval", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_decay", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--mlp_flag", type=int, default=1)
    parser.add_argument("--device", default='cuda:0')
    opt = parser.parse_args()
    model_run(opt)
