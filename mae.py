import pandas as pd
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from models_mae import mae_vit_base_patch16
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def segmentation(series, L, P):
    matrix_2D = series.reshape(L//P, P).T
    return matrix_2D

def normalization(Iraw, r):
    return ((Iraw - np.mean(Iraw))/np.std(Iraw)) * r
    
def renderization(Inorm):
    Igrey = np.stack([Inorm] * 3, axis=-1)
    return Igrey

def alignment(Igrey, L, H):
    S = 16 # patch size in pixels
    W = 224 # ImageNet pixels per size
    N = W//S # number of patches
    c = 0.4 # hyperparameter

    adjust_ratio = N*L/(L+H)
    Igrey_tensor = torch.from_numpy(Igrey).permute(2, 0, 1).float()
    n = int(c* adjust_ratio)
    resize = T.Resize((N*S, n*S), antialias=True)
    Igrey_resized = resize(Igrey_tensor)
    masked_part = torch.zeros(
        (3, N*S, (N-n)*S)
)
    
    final_image = torch.cat([Igrey_resized, masked_part], dim=-1)
    return final_image, N, n

def reconstruction(input_image, N, n, model):

    x = torch.unsqueeze(input_image, 0)
    mask = torch.ones((N, N))
    mask[:, :n] = torch.zeros((N, n))
    mask_ratio = torch.mean(mask).item()
    noise = mask.flatten().unsqueeze(0)
    with torch.no_grad():
        _, pred, mask = model(x, mask_ratio=mask_ratio, noise=noise)
    return model.unpatchify(pred).squeeze()

def reverse(img, series, L, H, P, r):
    img = torch.mean(img, dim=0, keepdim=True)  # (1, 224, 224)
    resize = T.Resize((P, (L+H)), antialias=True)
    img_resized = resize(img).squeeze(0)        # (P, W)

    denorm = (img_resized.numpy()/r)*np.std(series) + np.mean(series)

    ts = denorm.T.reshape(-1)
    return ts

def computeMetrics(original, predicted):
    mae = mean_absolute_error(original, predicted)
    mse = mean_squared_error(original, predicted)
    return mae, mse

datasets = [
    {
        "dataset_name" : "ETTh1",
        "url" : "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/refs/heads/main/ETT-small/ETTh1.csv",
        "P" : 24, # periodicity 
        "L" : 2880, # context length
        "r" : 0.4
    },

    {
        "dataset_name" : "ETTh2",
        "url" : "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/refs/heads/main/ETT-small/ETTh2.csv",
        "P" : 24,
        "L" : 1728,
        "r": 0.4
    },
    {
        "dataset_name" : "ETTm1",
        "url" : "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/refs/heads/main/ETT-small/ETTm1.csv",
        "P": 96,
        "L" : 2304,
        "r": 0.4
    },
    {
        "dataset_name" : "ETTm2",
        "url" : "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/refs/heads/main/ETT-small/ETTm2.csv",
        "P": 96,
        "L": 4032,
        "r": 0.4
    }
]



model = mae_vit_base_patch16()
ckpt = torch.load('mae_visualize_vit_base.pth', map_location='cpu')
_ = model.load_state_dict(ckpt['model'], strict=False)
model.eval()


H_values = [96, 192, 336, 720] # prediction length values

results = pd.DataFrame()
for dataset in datasets:
    df = pd.read_csv(dataset["url"])
    attributes = [col for col in df.columns if col not in ['date']]
    for H in H_values:
        for attr in attributes:
            L = dataset["L"]
            P = dataset["P"]
            r = dataset["r"]
            series = df[attr].values[:L]
            Iraw = segmentation(series,L,P)
            Inorm = normalization(Iraw, r)
            Igrey = renderization(Inorm)
            input_image, N, n = alignment(Igrey, L, H)

            plt.imshow(input_image.permute(1, 2, 0).numpy()[:, :, 0], cmap='gray')
            plt.title('Masked image')
            plt.savefig('image1')
            y = reconstruction(input_image, N, n, model)
            plt.imshow(y.permute(1,2,0).numpy()[:,:,0], cmap='gray')
            plt.title('Reconstructed image')
            ts = reverse(y, series,L,H, P, r)
            plt.axis('off')
            plt.savefig('image2')
            plt.clf()
            context_plus_horizon = df[attr].values[:(L+H)]
            mae, mse = computeMetrics(context_plus_horizon[-H:], ts[-H:])

           # plt.title(f"mae: {mae} mse: {mse}  {attr} - {dataset['dataset_name']} L: {L} H: {L}")
           # plt.plot(range(len(context_plus_horizon)),context_plus_horizon,label='actual')
           # plt.plot(range(L, len(ts)), ts[-H:], label='predicted')
           # plt.legend()
           # plt.savefig(f"prediction_{H}_{L}_{attr}_{dataset['dataset_name']}")
           # plt.clf()
            
            new_df = new_df = pd.DataFrame([{
                "mse": mae,
                "mae": mse,
                "attr": attr,
                "dataset": dataset['dataset_name'],
                "L": L,
                "H": H
            }])
            results = pd.concat([results, new_df], ignore_index=True)
            
#results.to_csv('results.csv', index=False)

df = results.groupby(['dataset', 'L', 'H'])[['mae','mse']].mean()
print(df)
