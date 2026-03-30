### this script is used to compare the performance of the different conformal prediction methods
## each conformal prediction will use different conformity scores
## the conformity scores will be calculated using the different methods
## 1. vanilla Score Function: use residual: |y - f(x)|
## 2. Feature Score in FCP: https://arxiv.org/pdf/2210.00173
       # use the feature vector v of the last layer of the model as the feature vector
       # let h = head function of the model: h(v) = y
       # S=inf_{v: h(v)=y} |v-\hat v| where \hat v is the feature vector of the last layer of the model with the input x
## 3. Feature Score in FFCP : https://arxiv.org/pdf/2412.00653 
       # use the taylor expansion of the head function of the model: h(v) = y
       # S=|y-f(x)|/||h'(\hat v)|| where h'(\hat v) is the gradient of the head function of the model at \hat v


from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
from torch.optim import SGD
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

## data loading from kagglehub

# path = kagglehub.dataset_download("murtozalikhon/brain-tumor-multimodal-image-ct-and-mri")
# if torch.cuda.is_available():
#     device = "cuda" 
# else:
#     device = "cpu" 
# #Resizing images and turning them into tensors for matrix operations
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

# #Creating Dataloaders for Magnetic Resonance Imaging
# MRI = datasets.ImageFolder(root=os.path.join(path, 'Dataset/Brain Tumor MRI images'), transform=transform)
# MRI_dataloader = DataLoader(MRI, batch_size=32, shuffle=True)
# MRI_train_size = int(0.4 * len(MRI))
# MRI_test_size = len(MRI) - MRI_train_size  
# MRI_train_dataset, MRI_test_dataset = random_split(MRI, [MRI_train_size, MRI_test_size])
# MRI_calib_size = int(0.8 * len(MRI_test_dataset))
# MRI_test_size = len(MRI_test_dataset) - MRI_calib_size  
# MRI_calib_dataset, MRI_test_dataset = random_split(MRI_test_dataset, [MRI_calib_size, MRI_test_size])

## save the data to the pt file
# torch.save(MRI, "MRI.pt")
# torch.save(MRI_train_dataset, "mri_train.pt")
# torch.save(MRI_calib_dataset, "mri_calib.pt")
# torch.save(MRI_test_dataset, "mri_test.pt")

## model training and evaluation
class MRIModelFCP(nn.Module):
    """
    MRI Model designed for FeatureCP with clear feature/head separation
    φ(x): feature extractor → R^128 feature space
    h(z): classification head → class probabilities
    """
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        
        # Feature extractor φ(x) - convolutional layers
        self.conv =nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(512, 1024, 3, 1, 1), nn.ReLU(), 
            nn.MaxPool2d(2), nn.MaxPool2d(4),
        )
        self.flatten = nn.Flatten()
        self.feature_layer = nn.Sequential(
            nn.Linear(7*7*1024, 256), nn.ReLU(), nn.Dropout(0.5)
        )  # φ(x) ∈ R^128 FEATURE SPACE
        # Classification head h(z)
        # Classification head h(z) - maps features to class probabilities
        self.classifier = nn.Linear(256, num_classes)
    
    def phi(self, x):
        """Feature extractor φ(x) → R^128"""
        x = self.conv(x)
        x = self.flatten(x)
        features = self.feature_layer(x)
        return features
    
    def h(self, z):
        """Classification head h(z) → class logits"""
        return self.classifier(z)
    
    def forward(self, x):
        """Full pipeline: h(φ(x))"""
        features = self.phi(x)
        return self.h(features)
    
    # # Initialize model
    # model = MRIModelFCP(in_channels=3)
    # model = model.to(device)


    # print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    # print(f"Classes: {MRI.classes}")

    # # Training function
    # def train_model(model, train_loader, num_epochs=10, lr=0.001):
    #     # Loss and optimizer
    #     criterion = nn.CrossEntropyLoss()
    #     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #     model.train()
    #     for epoch in range(num_epochs):
    #         running_loss = 0.0
    #         correct = 0
    #         total = 0
            
    #         for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
    #             images, labels = images.to(device), labels.to(device)
                
    #             optimizer.zero_grad()
    #             outputs = model(images)
    #             loss = criterion(outputs, labels)
    #             loss.backward()
    #             optimizer.step()
                
    #             running_loss += loss.item()
    #             _, predicted = torch.max(outputs.data, 1)
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()
            
    #         epoch_loss = running_loss / len(train_loader)
    #         epoch_acc = 100 * correct / total
    #         print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.2f}%")

    # # Train the model
    # print("Training MRI classification model...")
    # train_model(model, MRI_train_loader, num_epochs=10)

##### different score methods:
@torch.no_grad()
def vanilla_scores(model, loader, device="cuda"):
    device = device if (device=="cuda" and torch.cuda.is_available()) else "cpu"
    model.eval().to(device)
    S, Y = [], []
    V = []
    predicted_y = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        v = model.phi(xb).cpu().numpy()
        V.append(v)
        p = F.softmax(logits, dim=1)
        hat_y = torch.max(p, dim=1)[1]
        predicted_y.append(hat_y.cpu())
        # nonconformity: 1 - prob(true label)
        s = 1.0 - p[torch.arange(len(yb), device=device), yb]
        S.append(s.cpu()); Y.append(yb.cpu())
    predicted_y = torch.cat(predicted_y).numpy()
    return torch.cat(S).numpy(), torch.cat(Y).numpy(), np.vstack(V), predicted_y

def _surrogate_opt(model, z, y, steps=15, step_size=0.1, lam=1.0):
    """
    Minimize: CE(h(z_tilde), y) + lam * ||z_tilde - z||^2, starting at z_tilde=z.
    z: (B,d) **detached**. y: (B,)
    """
    # freeze model params to avoid computing/keeping their grads
    req_grads = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)

    try:
        z_tilde = z.clone().detach().requires_grad_(True)
        opt = SGD([z_tilde], lr=step_size, momentum=0.0)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            logits = model.h(z_tilde)                     # head only; params frozen
            ce = F.cross_entropy(logits, y)
            reg = ((z_tilde - z)**2).sum(dim=1).mean()
            (ce + lam*reg).backward()
            opt.step()
    finally:
        # restore original requires_grad flags
        for p, rg in zip(model.parameters(), req_grads):
            p.requires_grad_(rg)

    return z_tilde.detach()

def fcp_scores(model, loader, device="cuda", steps=15, step_size=0.1, lam=1.0):
    """
    Calibration scores s_i = ||z_tilde - z||_2 for true labels on the calibration set.
    """
    device = device if (device=="cuda" and torch.cuda.is_available()) else "cpu"
    model.eval().to(device)
    S, Y = [], []
    V = []
    predicted_y = []
    for xb, yb in loader:#tqdm(loader, desc="FCP scores"):
        xb, yb = xb.to(device), yb.to(device)
        with torch.no_grad():
            z = model.phi(xb).detach()                   
            v = z.cpu().numpy()
            V.append(v)
            hat_y = torch.max(model.h(z), dim=1)[1]
            predicted_y.append(hat_y.cpu())
        z_tilde = _surrogate_opt(model, z, yb, steps=steps, step_size=step_size, lam=lam)
        s = (z_tilde - z).norm(dim=1)
        S.append(s.cpu()); Y.append(yb.cpu())
    S = torch.cat(S).numpy()
    Y = torch.cat(Y).numpy()
    predicted_y = torch.cat(predicted_y).numpy()
    return S, Y, np.vstack(V), predicted_y

def fcp_predict_inner(model, loader, qhat, num_classes, device="cuda",
                      steps=15, step_size=0.1, lam=1.0):
    """
    Test-time feasibility: include class c iff ||z_tilde^c - z|| <= qhat.
    """
    device = device if (device=="cuda" and torch.cuda.is_available()) else "cpu"
    model.eval().to(device)
    all_sets, all_y = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device).long()
        with torch.no_grad():
            z = model.phi(xb).detach()
        B = z.shape[0]
        inc = torch.zeros((B, num_classes), dtype=torch.bool, device=device)
        for c in range(num_classes):
            yc = torch.full((B,), c, dtype=torch.long, device=device)
            z_tilde = _surrogate_opt(model, z, yc, steps=steps, step_size=step_size, lam=lam)
            d = (z_tilde - z).norm(dim=1)
            inc[:, c] = (d <= qhat)
        all_sets.append(inc.cpu()); all_y.append(yb.cpu())
    sets = torch.cat(all_sets, 0).numpy()
    y = torch.cat(all_y, 0).numpy()
    return sets, y

## FFCP
@torch.enable_grad()
def ffcp_scores(model, loader, device="cuda"):
    """return calibration score which ffcp will use split cp for the score.
    """
    device = device if (device=="cuda" and torch.cuda.is_available()) else "cpu"
    model.eval().to(device)
    S, Y = [], []
    V = []
    predicted_y = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        v = model.phi(xb).requires_grad_(True)          # (B,d)
        logits = model.h(v)                             # (B,C)
        V.append(v.cpu().numpy())
        predicted_y.append(torch.max(logits, dim=1)[1].cpu())
        B, C = logits.shape
        # margins for the true class
        true_logits = logits[torch.arange(B), yb]
        lmax_others, _ = torch.max(
            logits + torch.where(
                torch.arange(C, device=device)[None,:] == yb[:,None],
                torch.tensor(float("-inf"), device=device), 0.0
            ), dim=1
        )
        margin = true_logits - lmax_others              # (B,)
        g = torch.autograd.grad(true_logits.sum(), v, retain_graph=False)[0]  # (B,d)
        grad_norm = torch.linalg.norm(g, dim=1).clamp_min(1e-8)
        s = (-margin / grad_norm).clamp_min(0.0)        # larger = “farther” from decision
        S.append(s.detach().cpu()); Y.append(yb.cpu())
    predicted_y = torch.cat(predicted_y).numpy()
    return torch.cat(S).numpy(), torch.cat(Y).numpy(), np.vstack(V), predicted_y

@torch.enable_grad()
def ffcp_predict_sets(model, loader, qhat, device="cuda"):
    """Include class c if margin_c + qhat * ||grad logit_c|| >= 0."""
    device = device if (device=="cuda" and torch.cuda.is_available()) else "cpu"
    model.eval().to(device)
    all_sets, all_y = [], []
    for xb, yb in loader:#tqdm(loader, desc="FFCP predict sets"):
        xb, yb = xb.to(device), yb.to(device)
        v = model.phi(xb).requires_grad_(True)
        logits = model.h(v)
        B, C = logits.shape
        inc = torch.zeros((B, C), dtype=torch.bool, device=device)
        for c in range(C):
            lc = logits[:, c]
            mask = torch.ones(C, dtype=torch.bool, device=device); mask[c]=False
            lmax_others, _ = torch.max(logits[:, mask], dim=1)
            margin_c = lc - lmax_others
            g = torch.autograd.grad(lc.sum(), v, retain_graph=True)[0]
            gain = qhat * torch.linalg.norm(g, dim=1)
            inc[:, c] = (margin_c + gain) >= 0.0
        all_sets.append(inc.detach().cpu()); all_y.append(yb.cpu())
    return torch.cat(all_sets,0).numpy(), torch.cat(all_y,0).numpy()

def score_function(model, loader, device="cuda", score_type="vanilla"):
    device = device if torch.cuda.is_available() else "cpu"
    model.eval().to(device)

    if score_type == "vanilla":
        return vanilla_scores(model, loader, device)
    elif score_type == "fcp":
        return fcp_scores(model, loader, device)
    elif score_type == "ffcp":
        return ffcp_scores(model, loader, device)
    else:
        raise ValueError(f"Invalid score type: {score_type}")





##############################################################################################33
#### construct the conformal prediction model
import sys
import time

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# required or runnning conditional-conformal
os.environ["MOSEK_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

from FastKernCP.speedcp import SpeedCP
from FastKernCP.utils import *

# download conditional-conformal (Gibbs et al., 2023)
# !git clone https://github.com/jjcherian/conditional-conformal.git
from conditionalconformal import CondConf
from experiments.crossval import runCV

# download PCP (Zhang et al., 2004)
# !git clone https://github.com/yaozhang24/pcp.git
from PCP.utils import PCP, RLCP


@torch.no_grad()
def get_features(model, loader, device="cuda"):
    dev = device if (device=="cuda" and torch.cuda.is_available()) else "cpu"
    model.eval().to(dev)
    V = []
    for xb, yb in loader:
        xb = xb.to(dev)
        z = model.phi(xb).cpu().numpy()
        V.append(z)
    return np.vstack(V)


def split_cp(S_calib, S_test, alpha):
    start_time = time.time()
    n=len(S_calib)
    cutoffs = np.quantile(np.abs(S_calib), [(1 - alpha) * (1 + 1 / n)])[0]
    cutoffs = np.zeros(len(S_test)) + cutoffs
    coverages = (np.abs(S_test) <= cutoffs).astype(int)
    time_method = time.time()-start_time
    return cutoffs, coverages, time_method

def SpeedCP_threshold(X_cal, X_test, S_cal, S_test, Phi_cal, Phi_test,
                   alpha = 0.1, method ="randomized"):
  
    start_time = time.time()
    if method == "randomized":
        speedcp_algo = SpeedCP(
            alpha=alpha, max_steps=200, eps=1e-3, tol=1e-6, thres=10.0,
            ridge=1e-8, start_side='left', gamma=None,
            gamma_grid=np.logspace(0, 1, 20),
            randomize=True, verbose=False
        )
    elif method == "fixed":
        speedcp_algo = SpeedCP(
            alpha=alpha, max_steps=200, eps=1e-3, tol=1e-6, thres=10.0,
            ridge=1e-8, start_side='left', gamma=None,
            gamma_grid=np.logspace(0, 1, 20),
            randomize=False, verbose=False
        )
    elif method == "CV":
        speedcp_algo = SpeedCP(
            alpha=alpha, max_steps=200, eps=1e-3, tol=1e-6, thres=10.0,
            ridge=1e-8, start_side='left', gamma=None,
            gamma_grid=np.logspace(0, 1, 20), use_cv=True,
            randomize=True, verbose=False
        )
    else:
        raise ValueError(f"Invalid method: {method}")
    cutoffs, _ = speedcp_algo.fit(X_cal, Phi_cal, S_cal.ravel(), X_test, Phi_test)
    covers = (S_test <= cutoffs).astype(int)
    print(f"selected gamma: {speedcp_algo.gamma:.4f}, Selected lambda: {speedcp_algo.lam:.4f}")
    time_method = time.time()-start_time
    return cutoffs, covers, time_method


def get_output_dict(CP_method, alpha, S_calib, S_test, Xtrain_final, Xcalib_final, Xtest_final, Phi_calib, Phi_test, Y_calib):

    if CP_method == "Split" or CP_method == "all":
        print("Running SplitCP...")
        start_time = time.time()
        cutoffs, covers, time_method = split_cp(S_calib, S_test, alpha)
        method = "none" 
    elif CP_method == "RLCP":
        print("Running RLCP...")
        start_time = time.time()
        cutoffs, covers = RLCP(Xtrain_final, Xcalib_final, S_calib, Xtest_final, S_test, alpha, finite=True)
        time_method = time.time()-start_time
        method = "none"
        print("RLCP done")
    elif CP_method == "PCP":
        print("Running PCP...")
        start_time = time.time()
        pcp_algo = PCP()
        len_calib = len(Xcalib_final)
        rand_idx = np.random.permutation(len_calib)
        X_val = Xcalib_final[rand_idx[:int(len_calib*0.2)]]
        S_val = S_calib[rand_idx[:int(len_calib*0.2)]]
        X_calib_used = Xcalib_final[rand_idx[int(len_calib*0.2):]]
        S_calib_used = S_calib[rand_idx[int(len_calib*0.2):]]
        pcp_algo.train(X_val, S_val, info=True)
        cutoffs, covers = pcp_algo.calibrate(X_calib_used, S_calib_used, Xtest_final, S_test, alpha, finite=True)
        time_method = time.time()-start_time
        method = "none"
        print("PCP done")
    elif CP_method == "CondCP":
        print("Running CondCP...")
        start_time = time.time()
        # CondConf
        k = 5
        gamma = 4
        minRad = 0.0001
        maxRad = 1
        numRad = 40
        X_calib_used = np.hstack((Xcalib_final, Phi_calib))
        X_test_used = np.hstack((Xtest_final, Phi_test))
        phiFn=lambda x: x[:, Xcalib_final.shape[1]:]
        phiCalib = phiFn(X_calib_used)

        allLosses, radii = runCV(Xcalib_final, S_calib, 'rbf', gamma, alpha, k,
                                        minRad, maxRad, numRad, phiCalib)
        selectedRadius = radii[np.argmin(allLosses)]
        infinite_params = {'kernel': 'rbf', 'gamma': gamma, 'lambda': 1 / selectedRadius}

        # return 
        scoreFn = lambda x, y: x[:, -1]  # absolute residuals already computed
        condCovProgram = CondConf(score_fn = scoreFn, 
                                Phi_fn = phiFn, 
                                infinite_params = infinite_params)
        condCovProgram.setup_problem(X_calib_used, Y_calib.ravel(), S_calib.ravel())
        cutoffs=[]
        i=0
        for x, s in zip(X_test_used, S_test.ravel()):
            x_used=x.reshape(1,-1)
            cutoff = condCovProgram.predict(quantile=1-alpha,
                                        x_test=x_used,
                                        score_inv_fn = lambda s, x : [x - s, x + s],
                                        S_min=min(S_calib),
                                        S_max=max(S_calib),
                                        randomize=True,
                                        exact=False,
                                        threshold=1-alpha)
            cutoffs.append(np.abs(cutoff))
            i+=1
        covers = (np.abs(S_test) <= cutoffs).astype(int)
        cutoffs = np.array(cutoffs)
        time_method = time.time()-start_time
        method = "none"
        print("CondCP done")
    elif CP_method == "SpeedCP":
        method = "randomized"
    elif CP_method == "SpeedCP_fixed":
        method = "fixed"
    elif CP_method == "SpeedCP_CV":
        method = "CV"
    else:
        raise ValueError(f"Invalid SpeedCP method: {CP_method}")


    if method != "none":
        print("Running SpeedCP...")
        cutoffs, covers, time_method = SpeedCP_threshold(Xcalib_final, Xtest_final, S_calib, S_test, Phi_calib, Phi_test, alpha, method)
        print("SpeedCP done")


    return cutoffs, covers, time_method




import json

def cp_calibrate_and_eval(model, train_loader, calib_loader, test_loader, alpha=0.1, device="cuda", 
                    score_type="vanilla", use_pca=False, r_pca=8,
                    group_condition_type="intercept", n_groups = 0):
    # ---- define the score function
    try:
        S_calib, Y_calib, X_calib, predicted_y_calib = score_function(model, calib_loader, device, score_type)
        S_test, Y_test, X_test, predicted_y_test = score_function(model, test_loader, device, score_type)
        print("Score function computed")
    except:
        raise ValueError(f"Invalid score type: {score_type}")
    X_train = get_features(model, train_loader)
    # ---- define whether we need to use the PCA features
    scaler = StandardScaler().fit(X_train)
    if use_pca:
        print("Running PCA...")
        Xcalib_std = scaler.transform(X_calib)
        Xtest_std = scaler.transform(X_test)
        Xtrain_std = scaler.transform(X_train)
        pca = PCA(n_components=min(r_pca, Xcalib_std.shape[1]), random_state=0).fit(Xcalib_std)
        Xcalib_final = pca.transform(Xcalib_std)   # shape: (n_cal, r)
        Xtrain_final = pca.transform(Xtrain_std)
        Xtest_final = pca.transform(Xtest_std) # shape: (n_test, r)       
        print(f"PCA done")
    else:
        Xcalib_final = scaler.transform(X_calib)
        Xtest_final = scaler.transform(X_test)
        Xtrain_final = scaler.transform(X_train)
    
    ### define the conformal prediction method
    print("Constructing group condition...")
    if group_condition_type == "predicted_labels":
        model.eval()
        y_hat_calib = []
        with torch.no_grad():
            for xb, yb in calib_loader:#tqdm(calib_loader, desc="predicted labels calib"):
                xb = xb.to(device)
                logits = model(xb)
                p = F.softmax(logits, dim=1)
                pred = torch.argmax(p, dim=1)
                y_hat_calib.extend(pred.cpu().numpy())
        y_hat_calib = np.array(y_hat_calib)
        classes = np.unique(y_hat_calib)
        Phi_calib = np.zeros((len(y_hat_calib), len(classes)))
        # one-hot encoding
        Phi_calib[np.arange(len(y_hat_calib)), y_hat_calib] = 1
        # test set
        y_hat_test = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                p = F.softmax(logits, dim=1)
                pred = torch.argmax(p, dim=1)
                y_hat_test.extend(pred.cpu().numpy())
        y_hat_test = np.array(y_hat_test)
        Phi_test = np.zeros((len(y_hat_test), len(classes)))
        # one-hot encoding
        Phi_test[np.arange(len(y_hat_test)), y_hat_test] = 1
        # drop the first column for preventing the linear dependence
        Phi_calib = Phi_calib[:, 1:]
        Phi_test = Phi_test[:, 1:] 
        intercept_calib = np.ones((Xcalib_final.shape[0], 1))
        intercept_test = np.ones((Xtest_final.shape[0], 1))
        Phi_calib = np.concatenate([intercept_calib, Phi_calib], axis=1)
        Phi_test = np.concatenate([intercept_test, Phi_test], axis=1)
    elif group_condition_type == "Kmeans":
        from sklearn.cluster import KMeans
        print("Running Kmeans...")
        Xtest_all = Xtest_final
        #y_test_all = np.concatenate([run["y"] for run in runs])
        if Xtest_all.shape[1] < 2:
            raise ValueError("Xtest_all has less than 2 columns")
        km_function= KMeans(n_clusters=n_groups, random_state=0, n_init="auto")
        km_clusters = km_function.fit(Xcalib_final)
        Phi_calib = km_clusters.labels_ # shape: (n_cal, 1)
        # one-hot encoding
        Phi_calib = np.eye(n_groups)[Phi_calib] # shape: (n_cal, n_groups)
        # test set
        Phi_test = km_clusters.predict(Xtest_final)
        Phi_test = np.eye(n_groups)[Phi_test] # shape: (n_test, n_groups)
        # drop the first column for preventing the linear dependence
        Phi_calib = Phi_calib[:, 1:]
        Phi_test = Phi_test[:, 1:] 
        intercept_calib = np.ones((Xcalib_final.shape[0], 1))
        intercept_test = np.ones((Xtest_final.shape[0], 1))
        Phi_calib = np.concatenate([intercept_calib, Phi_calib], axis=1)
        Phi_test = np.concatenate([intercept_test, Phi_test], axis=1)
        #print(Phi_test.shape)
    else:
        Phi_calib = np.ones((Xcalib_final.shape[0], 1))
        Phi_test = np.ones((Xtest_final.shape[0], 1))
    print("Group condition constructed")
    
    return Xtrain_final, Xcalib_final, Xtest_final, Phi_calib, Phi_test, Y_calib, Y_test, S_calib, S_test, predicted_y_calib, predicted_y_test
    


import argparse
def main(CP_method, OUTPUT_DIR):

    parser = argparse.ArgumentParser(description="Run experiments with options")
    parser.add_argument("--group_condition_type", type=str, default="none") # none, Kmeans, predicted_labels, intercept
    parser.add_argument("--n_groups", type=int, default=0) # only for Kmeans
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--score_type", type=str, default="vanilla") # vanilla, fcp, ffcp
    parser.add_argument("--use_pca", type=bool, default=False)
    parser.add_argument("--r_pca", type=int, default=0)
    args = parser.parse_args()

    # load the data
    if torch.cuda.is_available():
        device = "cuda" 
    else:
        device = "cpu" 

    # load the data from the pt file
    MRI = torch.load("MRI.pt", weights_only=False)
    MRI_train_dataset = torch.load("mri_train.pt", weights_only=False)
    MRI_calib_dataset = torch.load("mri_calib.pt", weights_only=False)
    MRI_test_dataset  = torch.load("mri_test.pt", weights_only=False)
    from torch.utils.data import ConcatDataset, random_split
    MRI_combined = ConcatDataset([MRI_calib_dataset, MRI_test_dataset])
    # # create the data loader
    MRI_train_loader = DataLoader(MRI_train_dataset, batch_size=32, shuffle=True)
    # MRI_calib_loader = DataLoader(MRI_calib_dataset, batch_size=32, shuffle=True)
    # MRI_test_loader  = DataLoader(MRI_test_dataset, batch_size=32, shuffle=True)
    
    # load the model
    alpha = args.alpha
    model = MRIModelFCP(in_channels=3, num_classes=2)
    model.load_state_dict(torch.load("mri_model.pth", map_location=device))
    model = model.to(device)
    # run the experiments
    output_dir = os.path.join(OUTPUT_DIR, f"{args.score_type}_{args.group_condition_type}_{args.n_groups}_{args.use_pca}_{args.r_pca}")
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    multiple_runs = 50
    runs=[]
    if CP_method == "all":
        output_dir = os.path.join(output_dir, "all")
        os.makedirs(output_dir, exist_ok=True)
    if CP_method == "all":
        for i in range(multiple_runs):
            print(f"Running run {i}...")
            MRI_calib_dataset2, MRI_test_dataset2 = random_split(MRI_combined, [len(MRI_calib_dataset), len(MRI_test_dataset)])
            MRI_calib_loader = DataLoader(MRI_calib_dataset2, batch_size=32, shuffle=True)
            MRI_test_loader = DataLoader(MRI_test_dataset2, batch_size=32, shuffle=True)
            Xtrain_final, Xcalib_final, Xtest_final, Phi_calib, Phi_test, Y_calib, Y_test, S_calib, S_test, predicted_y_calib, predicted_y_test = cp_calibrate_and_eval(model, MRI_train_loader, MRI_calib_loader, MRI_test_loader, 
                                    args.alpha, device, args.score_type, args.use_pca, args.r_pca, 
                                    args.group_condition_type, args.n_groups)
            runs.append(i)
            data_info={"score_type": args.score_type, "use_pca": args.use_pca, "r_pca": args.r_pca, 
                        "group_condition_type": args.group_condition_type, "n_groups": args.n_groups, 
                        "alpha": args.alpha}
            results[f"data_info_json_{i}"] = np.array(json.dumps(data_info))
            results["runs"] = np.array(runs, dtype=np.int8)
            results[f"y_test_{i}"] = np.array(Y_test, dtype=np.int8)
            results[f"predicted_y_test_{i}"] = np.array(predicted_y_test, dtype=np.int8)
            results[f"predicted_y_calib_{i}"] = np.array(predicted_y_calib, dtype=np.int8)
            results[f"X_calib_{i}"] = np.array(Xcalib_final, dtype=np.float64)
            results[f"X_test_{i}"] = np.array(Xtest_final, dtype=np.float64)
            results[f"X_train_{i}"] = np.array(Xtrain_final, dtype=np.float64)
            results[f"y_calib_{i}"] = np.array(Y_calib, dtype=np.int8)
            results[f"S_calib_{i}"] = np.array(S_calib, dtype=np.float64)
            results[f"S_test_{i}"] = np.array(S_test, dtype=np.float64)
            for method_m in ["SpeedCP","SpeedCP_CV","SpeedCP_fixed","Split", "RLCP", "PCP"]:
                try:
                    cutoffs, covers, time_method = get_output_dict(method_m, alpha, S_calib, S_test, Xtrain_final, Xcalib_final, Xtest_final, Phi_calib, Phi_test, Y_calib)
                    qs = np.array(cutoffs, dtype=np.float64)
                    print(f"Cutoffs for {method_m} in run {i}: length {len(qs)}, value {np.median(qs)}")
                    cs = np.array(covers, dtype=np.int8)
                    print(f"Covers for {method_m} in run {i}: length {len(cs)}, value {np.sum(cs)/len(cs)}")
                    results[f"{method_m}_cutoffs_{i}"] = qs
                    results[f"{method_m}_covers_{i}"] = cs
                    results[f"{method_m}_time_{i}"] = time_method
                    output_file = os.path.join(output_dir, "all.npz")
                    np.savez(output_file, **results)
                    print(f"Saved results for {method_m} to {output_file}")
                except:
                    print(f"Error for {method_m}")
                    continue
           
    else:
        for i in range(multiple_runs):
            MRI_calib_dataset2, MRI_test_dataset2 = random_split(MRI_combined, [len(MRI_calib_dataset), len(MRI_test_dataset)])
            MRI_calib_loader = DataLoader(MRI_calib_dataset2, batch_size=32, shuffle=True)
            MRI_test_loader = DataLoader(MRI_test_dataset2, batch_size=32, shuffle=True)
            Xtrain_final, Xcalib_final, Xtest_final, Phi_calib, Phi_test, Y_calib, Y_test, S_calib, S_test, predicted_y_calib, predicted_y_test = cp_calibrate_and_eval(model, MRI_train_loader, MRI_calib_loader, MRI_test_loader, 
                                    args.alpha, device, args.score_type, args.use_pca, args.r_pca, 
                                    args.group_condition_type, args.n_groups)
            runs.append(i)
            data_info={"score_type": args.score_type, "use_pca": args.use_pca, "r_pca": args.r_pca, 
                        "group_condition_type": args.group_condition_type, "n_groups": args.n_groups, 
                        "alpha": args.alpha}
            results[f"data_info_json_{i}"] = np.array(json.dumps(data_info))
            results["runs"] = np.array(runs, dtype=np.int8)
            results[f"y_test_{i}"] = np.array(Y_test, dtype=np.int8)
            results[f"predicted_y_test_{i}"] = np.array(predicted_y_test, dtype=np.int8)
            results[f"predicted_y_calib_{i}"] = np.array(predicted_y_calib, dtype=np.int8)
            results[f"X_calib_{i}"] = np.array(Xcalib_final, dtype=np.float64)
            results[f"X_test_{i}"] = np.array(Xtest_final, dtype=np.float64)
            results[f"X_train_{i}"] = np.array(Xtrain_final, dtype=np.float64)
            results[f"y_calib_{i}"] = np.array(Y_calib, dtype=np.int8)
            results[f"S_calib_{i}"] = np.array(S_calib, dtype=np.float64)
            results[f"S_test_{i}"] = np.array(S_test, dtype=np.float64)
            cutoffs, covers, time_method = get_output_dict(CP_method, alpha, S_calib, S_test, Xtrain_final, Xcalib_final, Xtest_final, Phi_calib, Phi_test, Y_calib)
            qs = np.array(cutoffs, dtype=np.float64)
            cs = np.array(covers, dtype=np.int8)
            results[f"cutoffs_{i}"] = qs
            results[f"covers_{i}"] = cs
            results[f"time_{i}"] = time_method
            output_file = os.path.join(output_dir, f"{CP_method}.npz")
            np.savez(output_file, **results)
            print(f"Saved results for all methods to {output_file}")

# create the main function
if __name__ == "__main__":
    OUTPUT_DIR = "mri_results/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main("all", OUTPUT_DIR)
    print("Done")


