import argparse
import os
import copy
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from core import FSRCNN, AverageMeter, calc_psnr
from datasets import TrainDataset, EvalDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=8) # تقليل الحجم لراحة الجهاز
    parser.add_argument('--num-epochs', type=int, default=20)
    args = parser.parse_args()

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FSRCNN(scale_factor=args.scale).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, persistent_workers=True)
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    checkpoint_path = os.path.join(args.outputs_dir, f'best_x{args.scale}.pth')
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        
    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()
        
        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description(f'Epoch {epoch}/{args.num_epochs-1}')
            for data in train_dataloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                preds = model(inputs)
                loss = criterion(preds, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.update(loss.item(), len(inputs))
                t.set_postfix(loss=f'{epoch_losses.avg:.10f}')
                t.update(len(inputs))

        # تقييم النموذج
        model.eval()
        epoch_psnr = AverageMeter()
        for data in eval_dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device, non_blocking = True), labels.to(device, non_blocking = True)
            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)
            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        print(f'Eval PSNR: {epoch_psnr.avg:.2f}')

        if epoch_psnr.avg > best_psnr:
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, os.path.join(args.outputs_dir, f'best_x{args.scale}.pth'))