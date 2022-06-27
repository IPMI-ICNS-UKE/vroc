import os

import numpy as np
import torch
from tqdm import tqdm, trange

from vroc.models import AutoEncoder


class AutoencoderGym:
    def __init__(self, train_loader, val_loader, device, out_path):
        self.device = device
        self.out_path = out_path
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.model = AutoEncoder().to(device=self.device)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def workout(self, n_epochs=100, validation_epoch=5, intermediate_save=False):
        pbar = trange(1, n_epochs + 1)
        val_loss = np.NAN
        epoch_loss = np.NAN

        for epoch in pbar:
            pbar.set_description(
                f"epoch: {epoch} \ttrain loss: {epoch_loss:.3f} \tval loss: {val_loss:.3f}"
            )
            running_loss = self._train()
            epoch_loss = running_loss / len(self.train_loader)

            if epoch % validation_epoch == 0:
                val_loss = self._validation()

                if intermediate_save:
                    self._save_model(epoch=epoch, val_loss=val_loss)

        self._save_model(epoch=pbar[-1], val_loss=val_loss)

    def _validation(self):
        val_loss = 0.0
        for data, _ in self.val_loader:
            with torch.no_grad():
                images = data.to(self.device)
                outputs, embedded = self.model(images)
                outputs = torch.sigmoid(outputs)
                loss = self.criterion(outputs, images)
                val_loss += loss.item() * images.size(0)

        val_loss = val_loss / len(self.val_loader)
        return val_loss

    def _save_model(self, epoch, val_loss):
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(
            state,
            os.path.join(
                self.out_path, f"epoch{epoch:03d}_val_loss_=_{val_loss:.03f}.pth"
            ),
        )

    def _train(self):
        running_loss = 0.0
        for data, _ in self.train_loader:
            images = data.to(self.device)
            self.optimizer.zero_grad()
            outputs, _ = self.model(images)
            outputs = torch.sigmoid(outputs)
            loss = self.criterion(outputs, images)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * images.size(0)
        return running_loss
