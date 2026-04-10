# OCR from scratch (no deep learning libs). Plain numpy MLP + pygame UI.
# Trains on MNIST with random occlusion (masking a patch), then predicts on your drawn digits.
# Steps:
#   pip install pygame numpy
#   python ocr_partially.py --train           # trains, saves weights to partial_ocr_weights.npz
#   python ocr_partially.py --gui             # draw and predict (needs saved weights)
#
# Note: Numpy training is slower than torch. By default it uses a subset to keep it reasonable.

import argparse
import gzip
import os
import sys
import urllib.request
import numpy as np
import pygame


# MNIST URLs (cvdf mirror; original Yann site may 404)
MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


# ---------- MNIST loading (from scratch) ----------
def download_mnist(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    for name, url in MNIST_URLS.items():
        out = os.path.join(data_dir, url.split("/")[-1])
        if not os.path.exists(out):
            try:
                print(f"Downloading {url}")
                urllib.request.urlretrieve(url, out)
            except Exception as e:
                print(f"Download failed: {e}")
                print("If network is blocked, manually place the 4 MNIST *.gz files into", data_dir)
                raise


def parse_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        if magic != 2051:
            raise ValueError("Not an IDX image file")
        n = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(n, rows * cols).astype(np.float32) / 255.0
        return data


def parse_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        if magic != 2049:
            raise ValueError("Not an IDX label file")
        n = int.from_bytes(f.read(4), "big")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels


def load_mnist(data_dir="./data", subset=20000):
    download_mnist(data_dir)
    train_images = parse_idx_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
    train_labels = parse_idx_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    test_images = parse_idx_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
    test_labels = parse_idx_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))
    if subset and subset < len(train_images):
        train_images = train_images[:subset]
        train_labels = train_labels[:subset]
    return train_images, train_labels, test_images, test_labels


# ---------- simple MLP (numpy) ----------
class MLP:
    def __init__(self, input_dim=784, hidden=256, output=10):
        rng = np.random.default_rng(42)
        self.W1 = rng.normal(0, 0.1, (input_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 0.1, (hidden, output))
        self.b2 = np.zeros(output)
        self.last_lr = 0.01  # default lr for online updates

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        h1 = np.maximum(0, z1)  # ReLU
        z2 = h1 @ self.W2 + self.b2
        return z1, h1, z2

    def predict_logits(self, x):
        _, _, z2 = self.forward(x)
        return z2

    def train_one(self, x, label, lr=None):
        # one-step SGD on a single sample x (1x784) and int label
        if lr is None:
            lr = self.last_lr
        z1, h1, z2 = self.forward(x)
        probs = softmax(z2)
        n = x.shape[0]
        grad_z2 = probs
        grad_z2[np.arange(n), label] -= 1
        grad_z2 /= n
        grad_W2 = h1.T @ grad_z2
        grad_b2 = grad_z2.sum(axis=0)
        grad_h1 = grad_z2 @ self.W2.T
        grad_z1 = grad_h1 * (z1 > 0)
        grad_W1 = x.T @ grad_z1
        grad_b1 = grad_z1.sum(axis=0)
        self.W2 -= lr * grad_W2
        self.b2 -= lr * grad_b2
        self.W1 -= lr * grad_W1
        self.b1 -= lr * grad_b1

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load(self, path):
        data = np.load(path)
        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]


def softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def cross_entropy(probs, labels):
    n = labels.shape[0]
    return -np.log(probs[np.arange(n), labels] + 1e-9).mean()


def random_occlude(batch, max_frac=0.20):
    # batch shape: (N, 784). Mask some images with smaller patches so digits stay readable.
    N = batch.shape[0]
    imgs = batch.reshape(N, 28, 28).copy()
    for i in range(N):
        if np.random.rand() < 0.6:  # mask about 60% of images
            frac = np.random.uniform(0.08, max_frac)
            mh = max(2, int(28 * frac))
            mw = max(2, int(28 * frac))
            top = np.random.randint(0, 28 - mh + 1)
            left = np.random.randint(0, 28 - mw + 1)
            imgs[i, top : top + mh, left : left + mw] = 0.0
    return imgs.reshape(N, 784)


def accuracy(logits, labels):
    preds = logits.argmax(axis=1)
    return (preds == labels).mean() * 100.0


def train(args):
    x_train, y_train, x_test, y_test = load_mnist(subset=args.subset)
    model = MLP()
    if os.path.exists(args.model):
        # continue training from previous run so it keeps learning
        print(f"Loading existing weights from {args.model} to continue training...")
        model.load(args.model)
    model.last_lr = args.lr
    lr = args.lr
    batch = args.batch
    epochs = args.epochs
    print(f"Training on {len(x_train)} samples with occlusion...")
    for epoch in range(epochs):
        # shuffle
        idx = np.random.permutation(len(x_train))
        x_train = x_train[idx]
        y_train = y_train[idx]
        # mini-batches
        for start in range(0, len(x_train), batch):
            end = start + batch
            xb = x_train[start:end]
            yb = y_train[start:end]
            xb_occ = random_occlude(xb)  # apply mask only in training
            z1, h1, z2 = model.forward(xb_occ)
            probs = softmax(z2)
            loss = cross_entropy(probs, yb)
            # backprop (manual)
            n = xb.shape[0]
            grad_z2 = probs
            grad_z2[np.arange(n), yb] -= 1
            grad_z2 /= n
            grad_W2 = h1.T @ grad_z2
            grad_b2 = grad_z2.sum(axis=0)
            grad_h1 = grad_z2 @ model.W2.T
            grad_z1 = grad_h1 * (z1 > 0)
            grad_W1 = xb_occ.T @ grad_z1
            grad_b1 = grad_z1.sum(axis=0)
            # sgd step
            model.W2 -= lr * grad_W2
            model.b2 -= lr * grad_b2
            model.W1 -= lr * grad_W1
            model.b1 -= lr * grad_b1
        # eval each epoch
        train_acc = accuracy(model.predict_logits(x_train), y_train)
        test_acc = accuracy(model.predict_logits(x_test), y_test)
        print(f"Epoch {epoch+1}/{epochs} | train acc {train_acc:.1f}% | test acc {test_acc:.1f}% | loss {loss:.3f}")
    model.save(args.model)
    print(f"Saved weights to {args.model}")


# ---------- drawing + prediction ----------
class DrawGUI:
    def __init__(self, model_path="partial_ocr_weights.npz"):
        pygame.init()
        self.w, self.h = 420, 550
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("OCR (partial-trained) - numpy")
        self.canvas_rect = pygame.Rect(20, 20, 380, 380)
        self.brush = 12
        self.white = (240, 240, 240)
        self.black = (0, 0, 0)
        self.gray = (50, 50, 50)
        self.font = pygame.font.SysFont("consolas", 18)
        self.canvas = pygame.Surface((self.canvas_rect.w, self.canvas_rect.h))
        self.canvas.fill(self.black)
        self.model = MLP()
        try:
            self.model.load(model_path)
        except FileNotFoundError:
            print(f"Model file {model_path} not found. Train first.", file=sys.stderr)
            sys.exit(1)
        self.pred_text = "Draw a digit. Click Predict."
        self.model_path = model_path
        self.last_img = None  # store last preprocessed image
        # quiz buttons
        self.btn_correct = pygame.Rect(20, 480, 160, 40)
        self.btn_wrong = pygame.Rect(200, 480, 160, 40)
        self.quiz_mode = False
        self.quiz_total = 0
        self.quiz_correct = 0
        self.quiz_mode = False
        self.quiz_total = 0
        self.quiz_correct = 0

    def run(self):
        running = True
        drawing = False
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1 and self.canvas_rect.collidepoint(event.pos):
                        drawing = True
                        self.paint(event.pos)
                    elif event.button == 1:
                        self.check_buttons(event.pos)
                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    drawing = False
                if event.type == pygame.MOUSEMOTION and drawing:
                    self.paint(event.pos)
            self.draw_ui()
            pygame.display.flip()
        pygame.quit()

    def paint(self, pos):
        x = pos[0] - self.canvas_rect.x
        y = pos[1] - self.canvas_rect.y
        pygame.draw.circle(self.canvas, self.white, (x, y), self.brush)

    def clear(self):
        self.canvas.fill(self.black)
        self.pred_text = "Cleared. Draw again."

    def check_buttons(self, pos):
        predict_rect = pygame.Rect(20, 400, 160, 40)
        clear_rect = pygame.Rect(220, 400, 160, 40)
        if predict_rect.collidepoint(pos):
            self.predict()
        elif clear_rect.collidepoint(pos):
            self.clear()
        elif self.quiz_mode:
            if self.btn_correct.collidepoint(pos):
                self.mark_correct()
            elif self.btn_wrong.collidepoint(pos):
                self.mark_wrong()

    def predict(self):
        # grab canvas; keep strokes bright and background dark (no invert)
        arr = pygame.surfarray.array3d(self.canvas).astype(np.float32)
        gray = arr[:, :, 0] / 255.0  # 0..1, strokes near 1, background near 0
        # scale down to 28x28
        surf28 = pygame.transform.smoothscale(
            pygame.surfarray.make_surface(np.stack([gray * 255] * 3, axis=2).astype(np.uint8)),
            (28, 28),
        )
        img_arr = pygame.surfarray.array3d(surf28)[:, :, 0].T / 255.0  # transpose to match orientation
        img_arr = img_arr.reshape(1, 784)
        logits = self.model.predict_logits(img_arr)
        probs = softmax(logits)[0]
        top = probs.argmax()
        self.pred_text = f"Pred: {top} (p={probs[top]*100:.1f}%)"
        self.last_img = img_arr
        if self.quiz_mode:
            # waiting for user to mark correct/incorrect
            pass

    def draw_ui(self):
        self.screen.fill(self.gray)
        pygame.draw.rect(self.screen, (200, 200, 200), self.canvas_rect, 2)
        self.screen.blit(self.canvas, (self.canvas_rect.x, self.canvas_rect.y))
        predict_rect = pygame.Rect(20, 400, 160, 40)
        clear_rect = pygame.Rect(220, 400, 160, 40)
        pygame.draw.rect(self.screen, (80, 150, 255), predict_rect, border_radius=6)
        pygame.draw.rect(self.screen, (150, 80, 80), clear_rect, border_radius=6)
        self.screen.blit(self.font.render("Predict", True, self.black), (predict_rect.x + 40, predict_rect.y + 10))
        self.screen.blit(self.font.render("Clear", True, self.white), (clear_rect.x + 55, clear_rect.y + 10))
        self.screen.blit(self.font.render(self.pred_text, True, self.white), (20, 360))
        helper = "Trained on masked digits. Here you draw full digits."
        self.screen.blit(self.font.render(helper, True, self.white), (20, 430))
        if self.quiz_mode:
            quiz_msg = f"Quiz mode: {self.quiz_correct}/{self.quiz_total} correct. Mark after each predict."
            self.screen.blit(self.font.render(quiz_msg, True, self.white), (20, 450))
            pygame.draw.rect(self.screen, (90, 180, 120), self.btn_correct, border_radius=6)
            pygame.draw.rect(self.screen, (180, 90, 90), self.btn_wrong, border_radius=6)
            self.screen.blit(self.font.render("Correct", True, self.black), (self.btn_correct.x + 30, self.btn_correct.y + 10))
            self.screen.blit(self.font.render("Incorrect", True, self.white), (self.btn_wrong.x + 18, self.btn_wrong.y + 10))

    def mark_correct(self):
        if self.quiz_mode and self.last_img is not None:
            self.quiz_total += 1
            self.quiz_correct += 1
            self.pred_text = f"Marked correct. Score {self.quiz_correct}/{self.quiz_total}"

    def mark_wrong(self):
        if self.quiz_mode and self.last_img is not None:
            self.quiz_total += 1
            self.pred_text = f"Marked incorrect. Score {self.quiz_correct}/{self.quiz_total}"


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="OCR with partial MNIST (numpy, student-style)")
    parser.add_argument("--train", action="store_true", help="train the numpy MLP with occlusion")
    parser.add_argument("--epochs", type=int, default=3, help="epochs for training")
    parser.add_argument("--batch", type=int, default=128, help="batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--subset", type=int, default=20000, help="how many train samples to use (None for full)")
    parser.add_argument("--model", type=str, default="partial_ocr_weights.npz", help="where to save/load weights")
    parser.add_argument("--gui", action="store_true", help="launch drawing GUI")
    parser.add_argument("--clear", action="store_true", help="delete saved weights and exit")
    parser.add_argument("--quiz", action="store_true", help="enter quiz mode inside the GUI (manual correct/wrong)")
    args = parser.parse_args()

    if args.clear:
        if os.path.exists(args.model):
            try:
                os.remove(args.model)
                print(f"Deleted weights file {args.model}")
            except Exception as e:
                print(f"Could not delete {args.model}: {e}")
        else:
            print(f"No weights file {args.model} found.")
        return

    if args.train:
        train(args)
    if args.gui:
        gui = DrawGUI(model_path=args.model)
        gui.quiz_mode = args.quiz
        gui.run()
    if not args.train and not args.gui:
        print("Nothing to do. Use --train to train, --gui to draw, or --quiz with --gui.")


if __name__ == "__main__":
    main()
