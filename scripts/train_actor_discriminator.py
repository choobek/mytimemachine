import os
import json
import time
import random
import argparse
from types import SimpleNamespace

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader

import sys
sys.path.append('.')

from datasets.binary_identity_dataset import BinaryIdentityDataset
from models.binary_identity_model import build_identity_model


def set_seed(seed: int = 42):
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def build_transforms(input_size: int = 112):
	train_t = transforms.Compose([
		transforms.Resize((input_size, input_size)),
		transforms.RandomHorizontalFlip(0.5),
		transforms.ToTensor(),
		transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
	])
	val_t = transforms.Compose([
		transforms.Resize((input_size, input_size)),
		transforms.ToTensor(),
		transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
	])
	return train_t, val_t


def split_train_val_indices(samples, val_ratio: float = 0.1, seed: int = 42):
	pos_idx = [i for i, (_, y) in enumerate(samples) if y == 1]
	neg_idx = [i for i, (_, y) in enumerate(samples) if y == 0]
	rnd = random.Random(seed)
	rnd.shuffle(pos_idx)
	rnd.shuffle(neg_idx)
	n_pos_val = max(1, int(len(pos_idx) * val_ratio))
	n_neg_val = max(1, int(len(neg_idx) * val_ratio))
	val_indices = set(pos_idx[:n_pos_val] + neg_idx[:n_neg_val])
	train_indices = [i for i in range(len(samples)) if i not in val_indices]
	val_indices = list(sorted(val_indices))
	return train_indices, val_indices


def compute_class_weights(samples, indices):
	# weights for CrossEntropyLoss: tensor([w_neg, w_pos])
	pos = sum(1 for i in indices if samples[i][1] == 1)
	neg = sum(1 for i in indices if samples[i][1] == 0)
	total = pos + neg
	# inverse frequency normalized so average weight ~ 1
	w_pos = total / (2.0 * max(1, pos))
	w_neg = total / (2.0 * max(1, neg))
	return torch.tensor([w_neg, w_pos], dtype=torch.float32)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
	pred = logits.argmax(dim=1)
	correct = (pred == labels).sum().item()
	return correct / max(1, labels.size(0))


def train_one_epoch(model, loader, criterion, optimizer, device, limit_batches: int | None = None, log_interval: int = 200):
	model.train()
	running_loss = 0.0
	running_correct = 0
	running_count = 0
	for bi, (images, labels) in enumerate(loader):
		if limit_batches is not None and bi >= limit_batches:
			break
		images = images.to(device)
		labels = labels.to(device).long()
		optimizer.zero_grad(set_to_none=True)
		logits = model(images)
		loss = criterion(logits, labels)
		loss.backward()
		optimizer.step()
		with torch.no_grad():
			running_loss += loss.item() * labels.size(0)
			running_correct += (logits.argmax(dim=1) == labels).sum().item()
			running_count += labels.size(0)
			if log_interval and (bi % log_interval == 0):
				avg_loss = running_loss / max(1, running_count)
				avg_acc = running_correct / max(1, running_count)
				print(f"  [train] batch {bi}/{len(loader)}  avg_loss={avg_loss:.4f} avg_acc={avg_acc*100:.2f}%", flush=True)
	return running_loss / max(1, running_count), running_correct / max(1, running_count)


@torch.no_grad()
def evaluate(model, loader, criterion, device, limit_batches: int | None = None):
	model.eval()
	running_loss = 0.0
	running_correct = 0
	running_count = 0
	for bi, (images, labels) in enumerate(loader):
		if limit_batches is not None and bi >= limit_batches:
			break
		images = images.to(device)
		labels = labels.to(device).long()
		logits = model(images)
		loss = criterion(logits, labels)
		running_loss += loss.item() * labels.size(0)
		running_correct += (logits.argmax(dim=1) == labels).sum().item()
		running_count += labels.size(0)
	return running_loss / max(1, running_count), running_correct / max(1, running_count)


def save_json(path: str, data: dict):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, 'w') as f:
		json.dump(data, f, indent=2)


def main():
	parser = argparse.ArgumentParser(description='Train binary identity discriminator')
	parser.add_argument('--actor_root', type=str, default='data/all_130925')
	parser.add_argument('--non_actor_root', type=str, default='data/images1024x1024')
	parser.add_argument('--backend', type=str, default='arcface', choices=['arcface', 'resnet50', 'facenet'])
	parser.add_argument('--arcface_weights', type=str, default='pretrained_models/model_ir_se50.pth')
	parser.add_argument('--epochs', type=int, default=15)
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--val_ratio', type=float, default=0.1)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--input_size', type=int, default=112)
	parser.add_argument('--out_dir', type=str, default='experiments/full_training_run/actor_classifier')
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--limit_train_batches', type=int, default=0)
	parser.add_argument('--limit_val_batches', type=int, default=0)
	args = parser.parse_args()

	set_seed(args.seed)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# Transforms and datasets
	train_t, val_t = build_transforms(input_size=args.input_size)
	base_ds = BinaryIdentityDataset(actor_root=args.actor_root, non_actor_root=args.non_actor_root, transform=None)
	train_indices, val_indices = split_train_val_indices(base_ds.samples, val_ratio=args.val_ratio, seed=args.seed)
	train_ds = BinaryIdentityDataset(actor_root=args.actor_root, non_actor_root=args.non_actor_root, transform=train_t)
	val_ds = BinaryIdentityDataset(actor_root=args.actor_root, non_actor_root=args.non_actor_root, transform=val_t)
	train_subset = Subset(train_ds, train_indices)
	val_subset = Subset(val_ds, val_indices)

	# Report dataset sizes and class balance
	train_pos = sum(1 for i in train_indices if base_ds.samples[i][1] == 1)
	train_neg = sum(1 for i in train_indices if base_ds.samples[i][1] == 0)
	val_pos = sum(1 for i in val_indices if base_ds.samples[i][1] == 1)
	val_neg = sum(1 for i in val_indices if base_ds.samples[i][1] == 0)
	print(f"Train size: {len(train_subset)} (pos={train_pos}, neg={train_neg})", flush=True)
	print(f"Val size:   {len(val_subset)} (pos={val_pos}, neg={val_neg})", flush=True)

	# Dataloaders
	train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False, pin_memory=True)
	val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)

	# Model
	num_outputs = 2
	if args.backend == 'arcface':
		model = build_identity_model(backend='arcface', weights_path=args.arcface_weights, num_outputs=num_outputs, input_size=args.input_size)
	elif args.backend == 'resnet50':
		model = build_identity_model(backend='resnet50', weights_path=None, num_outputs=num_outputs, input_size=args.input_size)
	else:
		model = build_identity_model(backend='facenet', weights_path=None, num_outputs=num_outputs, input_size=args.input_size)
	model.to(device)

	# Loss with class weights to mitigate imbalance
	class_weights = compute_class_weights(base_ds.samples, train_indices).to(device)
	criterion = nn.CrossEntropyLoss(weight=class_weights)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

	# Prepare output dir and save options
	os.makedirs(args.out_dir, exist_ok=True)
	with open(os.path.join(args.out_dir, 'timestamp.txt'), 'w') as f:
		f.write(time.strftime('%Y-%m-%d %H:%M:%S'))
	save_json(os.path.join(args.out_dir, 'opt.json'), vars(args))

	best_val_acc = 0.0
	best_path = os.path.join(args.out_dir, 'actor_classifier_best.pth')
	final_path = os.path.join(args.out_dir, 'actor_classifier_final.pth')

	limit_train = args.limit_train_batches if args.limit_train_batches > 0 else None
	limit_val = args.limit_val_batches if args.limit_val_batches > 0 else None

	for epoch in range(1, args.epochs + 1):
		print(f"Starting epoch {epoch}/{args.epochs}...", flush=True)
		train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, limit_batches=limit_train, log_interval=200)
		val_loss, val_acc = evaluate(model, val_loader, criterion, device, limit_batches=limit_val)
		print(f'Epoch {epoch:03d}: train_loss={train_loss:.4f} acc={train_acc*100:.2f}% | val_loss={val_loss:.4f} acc={val_acc*100:.2f}%', flush=True)
		# Save best
		if val_acc > best_val_acc:
			best_val_acc = val_acc
			torch.save({'state_dict': model.state_dict(), 'val_acc': best_val_acc, 'epoch': epoch}, best_path)

	# Save final
	torch.save({'state_dict': model.state_dict(), 'best_val_acc': best_val_acc, 'epoch': args.epochs}, final_path)
	print(f'Saved best model to: {best_path}')
	print(f'Saved final model to: {final_path}')


if __name__ == '__main__':
	main()


